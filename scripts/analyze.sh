#!/bin/bash
# ============================================================================
# PRIMORDIAL V2 - Script di Analisi
# ============================================================================
# Uso: ./scripts/analyze.sh [opzioni]
#
# Opzioni:
#   -o FILE    Salva output su file (default: stdout)
#   -q         Modalità quiet (solo risultati, no header)
#   -h         Mostra help
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="$SCRIPT_DIR/analyze_simulation.sql"

# Database config
DB_HOST="${PGHOST:-localhost}"
DB_NAME="${PGDATABASE:-primordial_v2}"
DB_USER="${PGUSER:-primordial}"
DB_PASS="${PGPASSWORD:-primordial}"

# Parse arguments
OUTPUT=""
QUIET=""

while getopts "o:qh" opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        q) QUIET="-q" ;;
        h)
            echo "Uso: $0 [-o output_file] [-q] [-h]"
            echo ""
            echo "Opzioni:"
            echo "  -o FILE    Salva output su file"
            echo "  -q         Modalità quiet"
            echo "  -h         Mostra questo help"
            exit 0
            ;;
        *)
            echo "Opzione non valida: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Check SQL file exists
if [ ! -f "$SQL_FILE" ]; then
    echo "Errore: $SQL_FILE non trovato" >&2
    exit 1
fi

# Run analysis
if [ -n "$OUTPUT" ]; then
    echo "Esecuzione analisi... Output su: $OUTPUT"
    PGPASSWORD="$DB_PASS" psql -U "$DB_USER" -h "$DB_HOST" -d "$DB_NAME" $QUIET -f "$SQL_FILE" > "$OUTPUT" 2>&1
    echo "Analisi completata. Risultati salvati in: $OUTPUT"
else
    PGPASSWORD="$DB_PASS" psql -U "$DB_USER" -h "$DB_HOST" -d "$DB_NAME" $QUIET -f "$SQL_FILE"
fi
