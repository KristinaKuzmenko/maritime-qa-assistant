# =============================================================================
# Maritime QA - Infrastructure (EKS, ECR, IAM)
# =============================================================================
# This can be destroyed and recreated without losing data!
# S3 buckets are managed separately in ../storage/
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0, < 6.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.20"
    }
  }

  # ОТДЕЛЬНЫЙ STATE для инфраструктуры!
  backend "s3" {
    bucket  = "mlops-tfstate-kristinakuzmenko"
    key     = "maritime-qa/infrastructure/terraform.tfstate" # ← Отдельный ключ!
    region  = "eu-central-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Component   = "infrastructure"
    }
  }
}

data "aws_caller_identity" "current" {}

# =============================================================================
# Remote State - Get S3 bucket info from storage module
# =============================================================================

data "terraform_remote_state" "storage" {
  backend = "s3"

  config = {
    bucket = "mlops-tfstate-kristinakuzmenko"
    key    = "maritime-qa/storage-use1/terraform.tfstate"
    region = "eu-central-1"
  }
}

# Local values from storage
locals {
  s3_app_bucket_arn     = data.terraform_remote_state.storage.outputs.app_files_bucket_arn
  s3_app_bucket_name    = data.terraform_remote_state.storage.outputs.app_files_bucket_name
  s3_backup_bucket_arn  = data.terraform_remote_state.storage.outputs.qdrant_backups_bucket_arn
  s3_backup_bucket_name = data.terraform_remote_state.storage.outputs.qdrant_backups_bucket_name
  s3_access_policy_arn  = data.terraform_remote_state.storage.outputs.s3_access_policy_arn

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# =============================================================================
# VPC Module
# =============================================================================

module "vpc" {
  source = "../modules/vpc"

  vpc_name              = "${var.project_name}-${var.environment}-vpc"
  vpc_cidr              = var.vpc_cidr
  cluster_name          = var.cluster_name
  availability_zone     = var.availability_zone
  availability_zone_2   = var.availability_zone_2
  public_subnet_cidr    = var.public_subnet_cidr
  public_subnet_cidr_2  = var.public_subnet_cidr_2
  private_subnet_cidr   = var.private_subnet_cidr
  private_subnet_cidr_2 = var.private_subnet_cidr_2
  enable_nat_gateway    = true

  tags = local.common_tags
}

# =============================================================================
# EKS Module (with OIDC for IRSA)
# =============================================================================

module "eks" {
  source = "../modules/eks"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  node_instance_types = var.node_instance_types
  node_desired_size   = var.node_desired_size
  node_min_size       = var.node_min_size
  node_max_size       = var.node_max_size
  node_disk_size      = var.node_disk_size
  use_spot_instances  = var.use_spot_instances

  tags = local.common_tags
}

# =============================================================================
# ECR Repositories
# =============================================================================

module "ecr" {
  source = "../modules/ecr"

  project_name       = var.project_name
  create_qdrant_repo = false
  keep_image_count   = 10
  eks_node_role_arn  = module.eks.node_role_arn

  tags = local.common_tags
}

# =============================================================================
# IAM Role for App (IRSA) - uses S3 policy from storage module
# =============================================================================

resource "aws_iam_role" "app" {
  name = "${var.cluster_name}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider_url}:aud" = "sts.amazonaws.com"
          "${module.eks.oidc_provider_url}:sub" = "system:serviceaccount:${var.app_namespace}:${var.app_service_account}"
        }
      }
    }]
  })

  tags = local.common_tags
}

# Attach S3 policy from storage module
resource "aws_iam_role_policy_attachment" "app_s3" {
  role       = aws_iam_role.app.name
  policy_arn = local.s3_access_policy_arn
}

# CloudWatch Logs policy
resource "aws_iam_role_policy" "app_cloudwatch" {
  name = "cloudwatch-logs"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ]
      Resource = "arn:aws:logs:${var.aws_region}:*:log-group:/aws/eks/${var.cluster_name}/*"
    }]
  })
}

# =============================================================================
# IAM Role for Qdrant (for S3 backups)
# =============================================================================

resource "aws_iam_role" "qdrant" {
  name = "${var.cluster_name}-qdrant-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider_url}:aud" = "sts.amazonaws.com"
          "${module.eks.oidc_provider_url}:sub" = "system:serviceaccount:${var.app_namespace}:qdrant"
        }
      }
    }]
  })

  tags = local.common_tags
}

# =============================================================================
# External Secrets Operator (ESO) - IRSA role to read AWS Secrets Manager
# =============================================================================

resource "aws_iam_role" "external_secrets" {
  name = "${var.cluster_name}-external-secrets-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider_url}:aud" = "sts.amazonaws.com"
          "${module.eks.oidc_provider_url}:sub" = "system:serviceaccount:${var.external_secrets_namespace}:${var.external_secrets_service_account}"
        }
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "external_secrets" {
  name        = "${var.project_name}-external-secrets-${var.environment}-${var.aws_region}"
  description = "Allow External Secrets Operator to read AWS Secrets Manager secrets for Maritime QA"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSecretsManager"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/*"
        ]
      },
      {
        Sid    = "KmsDecryptForSecretsManager"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${var.aws_region}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "external_secrets" {
  role       = aws_iam_role.external_secrets.name
  policy_arn = aws_iam_policy.external_secrets.arn
}

# Qdrant S3 backup policy
resource "aws_iam_role_policy" "qdrant_s3_backup" {
  name = "s3-backup"
  role = aws_iam_role.qdrant.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        local.s3_backup_bucket_arn,
        "${local.s3_backup_bucket_arn}/*"
      ]
    }]
  })
}

# =============================================================================
# Security Group for Qdrant
# =============================================================================

resource "aws_security_group" "qdrant" {
  name        = "${var.project_name}-qdrant-sg"
  description = "Security group for Qdrant"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "Qdrant HTTP"
    from_port       = 6333
    to_port         = 6333
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  ingress {
    description     = "Qdrant gRPC"
    from_port       = 6334
    to_port         = 6334
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-qdrant-sg"
  })
}

# =============================================================================
# Generate Kubernetes Manifests
# =============================================================================

resource "local_file" "k8s_manifests" {
  filename = "${path.module}/generated/k8s-config.yaml"
  content  = <<-EOT
# =============================================================================
# Generated Kubernetes Configuration
# =============================================================================
# Apply with: kubectl apply -f generated/k8s-config.yaml
# =============================================================================

# Namespace for the application
apiVersion: v1
kind: Namespace
metadata:
  name: ${var.app_namespace}
---
# Namespace for External Secrets Operator
apiVersion: v1
kind: Namespace
metadata:
  name: ${var.external_secrets_namespace}

---
# ServiceAccount for External Secrets Operator (IRSA)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${var.external_secrets_service_account}
  namespace: ${var.external_secrets_namespace}
  annotations:
    eks.amazonaws.com/role-arn: ${aws_iam_role.external_secrets.arn}

---
# StorageClass for Qdrant PVC
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  fsType: ext4
  encrypted: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# ServiceAccount for App (IRSA)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${var.app_service_account}
  namespace: ${var.app_namespace}
  annotations:
    eks.amazonaws.com/role-arn: ${aws_iam_role.app.arn}

---
# ServiceAccount for Qdrant (IRSA for S3 backups)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: qdrant
  namespace: ${var.app_namespace}
  annotations:
    eks.amazonaws.com/role-arn: ${aws_iam_role.qdrant.arn}

---
# ConfigMap with S3 configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: maritime-qa-config
  namespace: ${var.app_namespace}
data:
  STORAGE_TYPE: "s3"
  S3_BUCKET_NAME: "${local.s3_app_bucket_name}"
  S3_BACKUP_BUCKET: "${local.s3_backup_bucket_name}"
  AWS_REGION: "${var.aws_region}"
  QDRANT_HOST: "${var.qdrant_host != "" ? var.qdrant_host : "qdrant.${var.app_namespace}.svc.cluster.local"}"
  QDRANT_PORT: "${var.qdrant_port}"
EOT
}
