# =============================================================================
# ECR Module - Container Registry for Maritime QA Application
# =============================================================================
# Repositories: app (single image), (optional) custom Qdrant
# Features: scanning, lifecycle policies, encryption
# =============================================================================

# -----------------------------------------------------------------------------
# Backend Application Repository
# -----------------------------------------------------------------------------

resource "aws_ecr_repository" "backend" {
  name                 = "${var.project_name}-app"
  image_tag_mutability = var.image_tag_mutability

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  lifecycle {
    prevent_destroy = true
  }

  tags = merge(var.tags, {
    Name    = "${var.project_name}-app"
    Service = "app"
  })
}

# -----------------------------------------------------------------------------
# Qdrant Repository (optional - if using custom image)
# -----------------------------------------------------------------------------

resource "aws_ecr_repository" "qdrant" {
  count                = var.create_qdrant_repo ? 1 : 0
  name                 = "${var.project_name}-qdrant"
  image_tag_mutability = var.image_tag_mutability

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(var.tags, {
    Name    = "${var.project_name}-qdrant"
    Service = "qdrant"
  })
}

# -----------------------------------------------------------------------------
# Lifecycle Policies - Clean up old images
# -----------------------------------------------------------------------------

resource "aws_ecr_lifecycle_policy" "backend" {
  repository = aws_ecr_repository.backend.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last ${var.keep_image_count} images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "release"]
          countType     = "imageCountMoreThan"
          countNumber   = var.keep_image_count
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Remove untagged images older than ${var.untagged_image_days} days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = var.untagged_image_days
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 3
        description  = "Keep last ${var.keep_image_count} of any tagged images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.keep_image_count * 2
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "qdrant" {
  count      = var.create_qdrant_repo ? 1 : 0
  repository = aws_ecr_repository.qdrant[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last ${var.keep_image_count} images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.keep_image_count
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Repository Policy - Allow EKS nodes to pull images
# -----------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

resource "aws_ecr_repository_policy" "backend" {
  repository = aws_ecr_repository.backend.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEKSPull"
        Effect = "Allow"
        Principal = {
          AWS = var.eks_node_role_arn
        }
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
      }
    ]
  })
}
