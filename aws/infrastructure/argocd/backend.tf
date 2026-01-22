terraform {
 backend "s3" {
  bucket = "mlops-tfstate-kristinakuzmenko"
  key   = "argocd/terraform.tfstate"
  region = "eu-central-1"
 }
}