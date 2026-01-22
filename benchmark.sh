#!/bin/bash
# Maritime QA Benchmark Runner
# Usage: ./benchmark.sh [load|real|clean]

set -e

case "${1:-help}" in
  load)
    echo "🚀 Running LOAD tests with LOCAL Qdrant (FREE)"
    echo ""
    echo "🔨 Building fresh image..."
    docker-compose -f docker-compose.benchmark.yml --profile load build
    
    # Create reports directory if it doesn't exist
    mkdir -p reports
    REPORT_FILE="reports/load_test_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "📝 Saving report to: $REPORT_FILE"
    docker-compose -f docker-compose.benchmark.yml --profile load up --abort-on-container-exit 2>&1 | tee "$REPORT_FILE"
    docker-compose -f docker-compose.benchmark.yml --profile load down
    
    echo ""
    echo "✅ Report saved to: $REPORT_FILE"
    ;;
    
  real)
    echo "⚠️  Running REAL tests with CLOUD services (COSTS MONEY)"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "🔨 Building fresh image (no cache)..."
      docker-compose -f docker-compose.benchmark.yml --profile real build --no-cache benchmark-real
      
      # Create reports directory if it doesn't exist
      mkdir -p reports
      REPORT_FILE="reports/real_test_$(date +%Y%m%d_%H%M%S).txt"
      
      echo "📝 Saving report to: $REPORT_FILE"
      docker-compose -f docker-compose.benchmark.yml --profile real run --rm benchmark-real \
        pytest backend/tests/benchmark_real.py -v -s 2>&1 | tee "$REPORT_FILE"
      
      echo ""
      echo "✅ Report saved to: $REPORT_FILE"
    else
      echo "Cancelled."
      exit 1
    fi
    ;;
    
  real-quick)
    echo "⚡ Quick cloud connection test (~$0.0001)"
    echo ""
    echo "🔨 Building fresh image..."
    docker-compose -f docker-compose.benchmark.yml --profile real build benchmark-real
    docker-compose -f docker-compose.benchmark.yml --profile real run --rm benchmark-real \
      pytest backend/tests/benchmark_real.py::test_real_cloud_services_connection -v -s
    ;;
    
  clean)
    echo "🧹 Cleaning up benchmark resources..."
    docker-compose -f docker-compose.benchmark.yml --profile load down -v
    docker-compose -f docker-compose.benchmark.yml --profile real down -v
    echo "✅ Cleanup complete"
    ;;
    
  *)
    echo "Maritime QA Benchmark Runner"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  load        Run load tests with local Qdrant (FREE, ~1 min)"
    echo "  real        Run real API tests with cloud services (EXPENSIVE, ~$1-10)"
    echo "  real-quick  Quick cloud connection test (cheap, ~$0.0001)"
    echo "  clean       Clean up all benchmark containers and volumes"
    echo ""
    echo "Examples:"
    echo "  $0 load              # Infrastructure load testing"
    echo "  $0 real-quick        # Verify cloud connections"
    echo "  $0 real              # Full production validation"
    echo "  $0 clean             # Remove containers and volumes"
    ;;
esac
