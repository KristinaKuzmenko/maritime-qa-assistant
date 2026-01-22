# =============================================================================
# IAM Module - IRSA for EKS Pods
# =============================================================================
# IAM Roles for Service Accounts (IRSA) - allows K8s pods to access AWS services
# Roles: App (S3 access), Qdrant (optional EBS snapshots)
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# App Role - S3 Access for Backend Pods
# -----------------------------------------------------------------------------

resource "aws_iam_role" "app" {
  name = "${var.cluster_name}-app-irsa-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.oidc_provider_arn, "/^(.*provider/)/", "")}:aud" = "sts.amazonaws.com"
            "${replace(var.oidc_provider_arn, "/^(.*provider/)/", "")}:sub" = "system:serviceaccount:${var.app_namespace}:${var.app_service_account}"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.cluster_name}-app-irsa-role"
    Service = "backend"
  })
}

# S3 Access Policy
resource "aws_iam_role_policy" "app_s3_access" {
  name = "s3-access"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation"
        ]
        Resource = [
          var.s3_bucket_arn,
          "${var.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

# CloudWatch Logs Policy (for application logging)
resource "aws_iam_role_policy" "app_cloudwatch" {
  count = var.enable_cloudwatch_logs ? 1 : 0
  name  = "cloudwatch-logs"
  role  = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/eks/${var.cluster_name}/*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Qdrant Role - EBS Snapshots (optional, for backup)
# -----------------------------------------------------------------------------

resource "aws_iam_role" "qdrant" {
  count = var.create_qdrant_role ? 1 : 0
  name  = "${var.cluster_name}-qdrant-irsa-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.oidc_provider_arn, "/^(.*provider/)/", "")}:aud" = "sts.amazonaws.com"
            "${replace(var.oidc_provider_arn, "/^(.*provider/)/", "")}:sub" = "system:serviceaccount:${var.app_namespace}:${var.qdrant_service_account}"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.cluster_name}-qdrant-irsa-role"
    Service = "qdrant"
  })
}

# EBS Snapshot Policy for Qdrant backups
resource "aws_iam_role_policy" "qdrant_ebs_snapshots" {
  count = var.create_qdrant_role ? 1 : 0
  name  = "ebs-snapshots"
  role  = aws_iam_role.qdrant[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EBSSnapshots"
        Effect = "Allow"
        Action = [
          "ec2:CreateSnapshot",
          "ec2:DeleteSnapshot",
          "ec2:DescribeSnapshots",
          "ec2:DescribeVolumes",
          "ec2:CreateTags"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/kubernetes.io/cluster/${var.cluster_name}" = "owned"
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# EKS Node Role Additional Policies
# -----------------------------------------------------------------------------

# Policy for nodes to pull from ECR
resource "aws_iam_role_policy" "node_ecr_access" {
  count = var.node_role_name != "" ? 1 : 0
  name  = "ecr-pull-access"
  role  = var.node_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Kubernetes Service Account Annotation Helper
# -----------------------------------------------------------------------------

# This outputs the annotation needed for the K8s ServiceAccount
output "app_service_account_annotation" {
  description = "Annotation to add to Kubernetes ServiceAccount"
  value = {
    "eks.amazonaws.com/role-arn" = aws_iam_role.app.arn
  }
}
