resource "kubernetes_namespace" "argo" {
  metadata { name = var.argocd_namespace } 
}

# Argo CD (ApplicationSet is disabled in values.yaml)
resource "helm_release" "argo" {
  name       = "argocd"
  namespace  = kubernetes_namespace.argo.metadata[0].name
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-cd"
  version    = var.argocd_chart_version

  wait    = true
  atomic  = true
  timeout = 600

  values = [file("${path.module}/values/argocd-values.yaml")]
}

#  ApplicationSet CRD + controller 
resource "helm_release" "appset" {
  name       = "argocd-appset"
  namespace  = kubernetes_namespace.argo.metadata[0].name
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argocd-applicationset"
  version    = "1.12.1"

  wait    = true
  atomic  = true
  timeout = 600

  values = [yamlencode({
    installCRDs = true
    rbac = { pspEnabled = false }
    podSecurityPolicy = { enabled = false }
  })]

  depends_on = [helm_release.argo]
}

# =============================================================================
# Argo CD Application (deploy this repo out-of-the-box)
# NOTE: Commented out to avoid CRD chicken-egg problem.
# Will be created manually after ArgoCD is installed.
# =============================================================================

# resource "kubernetes_manifest" "maritime_qa_app" {
#   manifest = {
#     apiVersion = "argoproj.io/v1alpha1"
#     kind       = "Application"
#     metadata = {
#       name      = var.argocd_app_name
#       namespace = kubernetes_namespace.argo.metadata[0].name
#     }
#     spec = {
#       project = "default"
#       source = {
#         repoURL        = var.app_repo_url
#         targetRevision = var.app_repo_branch
#         path           = var.app_kustomize_path
#       }
#       destination = {
#         server    = "https://kubernetes.default.svc"
#         namespace = var.app_destination_namespace
#       }
#       syncPolicy = {
#         automated = {
#           prune    = true
#           selfHeal = true
#         }
#         syncOptions = [
#           "CreateNamespace=true",
#         ]
#       }
#     }
#   }
#
#   depends_on = [
#     helm_release.argo,
#   ]
# }
