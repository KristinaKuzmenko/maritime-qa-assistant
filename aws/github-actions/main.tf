terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket  = "maritime-qa-terraform-state"
    key     = "github-actions/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
}

# ============================================================================
# GitHub OIDC Provider for AWS IAM
# ============================================================================
# Allows GitHub Actions to assume IAM roles without storing credentials

resource "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
  
  client_id_list = [
    "sts.amazonaws.com"
  ]
  
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",  # GitHub's OIDC thumbprint
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd"   # Backup thumbprint
  ]
  
  tags = {
    Name        = "github-actions-oidc"
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# IAM Role for GitHub Actions - ECR Push
# ============================================================================
# Role that GitHub Actions workflows assume to push Docker images to ECR

data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    effect = "Allow"
    
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }
    
    actions = ["sts:AssumeRoleWithWebIdentity"]
    
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_org}/${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions_ecr_push" {
  name               = "${var.project_name}-${var.environment}-github-actions-ecr-push"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json
  description        = "Role for GitHub Actions to push Docker images to ECR"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-github-actions-ecr-push"
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# IAM Policy for ECR Push
# ============================================================================

data "aws_iam_policy_document" "ecr_push_policy" {
  # ECR authentication
  statement {
    sid    = "ECRGetAuthorizationToken"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken"
    ]
    resources = ["*"]
  }
  
  # ECR repository operations
  statement {
    sid    = "ECRRepositoryAccess"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:DescribeRepositories",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:ListImages",
      "ecr:PutImage",
      "ecr:UploadLayerPart"
    ]
    resources = [
      "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/${var.ecr_repository_name}"
    ]
  }
}

resource "aws_iam_role_policy" "github_actions_ecr_push" {
  name   = "ECRPushPolicy"
  role   = aws_iam_role.github_actions_ecr_push.id
  policy = data.aws_iam_policy_document.ecr_push_policy.json
}

# ============================================================================
# Data Sources
# ============================================================================

data "aws_caller_identity" "current" {}

# ============================================================================
# Outputs
# ============================================================================

output "github_actions_role_arn" {
  description = "ARN of the IAM role for GitHub Actions"
  value       = aws_iam_role.github_actions_ecr_push.arn
}

output "github_oidc_provider_arn" {
  description = "ARN of the GitHub OIDC provider"
  value       = aws_iam_openid_connect_provider.github_actions.arn
}

output "instructions" {
  description = "Instructions for using this role in GitHub Actions"
  value = <<-EOT
    
    GitHub Actions Role Created Successfully!
    
    Role ARN: ${aws_iam_role.github_actions_ecr_push.arn}
    
    Add this role ARN to your GitHub Actions workflow:
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        role-to-assume: ${aws_iam_role.github_actions_ecr_push.arn}
        aws-region: ${var.aws_region}
    
    No secrets needed in GitHub repository - uses OIDC authentication!
  EOT
}
