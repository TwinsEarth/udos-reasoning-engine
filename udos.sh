#!/bin/bash
# 根目录便捷入口（macOS/Linux）：在仓库根直接运行  ./udos.sh <命令>
# 例如：./udos.sh info / ./udos.sh serve / ./udos.sh health
DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
exec "$DIR/bin/udos" "$@"
