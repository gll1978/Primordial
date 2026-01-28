#!/bin/bash
# Setup PostgreSQL database for PRIMORDIAL V2
# Usage: ./scripts/setup_db.sh [DB_USER] [DB_NAME]

set -e

DB_USER="${1:-primordial}"
DB_NAME="${2:-primordial_v2}"
DB_PASS="${DB_PASS:-primordial}"
SCHEMA_FILE="$(dirname "$0")/../schema.sql"

echo "=== PRIMORDIAL V2 Database Setup ==="
echo "User: $DB_USER"
echo "Database: $DB_NAME"
echo ""

# Create user if it doesn't exist
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
    echo "Creating user '$DB_USER'..."
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
else
    echo "User '$DB_USER' already exists."
fi

# Create database if it doesn't exist
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
    echo "Creating database '$DB_NAME'..."
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
else
    echo "Database '$DB_NAME' already exists."
fi

# Grant privileges
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Apply schema
echo "Applying schema..."
PGPASSWORD="$DB_PASS" psql -U "$DB_USER" -d "$DB_NAME" -f "$SCHEMA_FILE"

echo ""
echo "=== Setup complete ==="
echo "Connection URL: postgresql://$DB_USER:$DB_PASS@localhost/$DB_NAME"
echo ""
echo "Test with: cargo run --features database -- run --steps 1000 --database-url \"postgresql://$DB_USER:$DB_PASS@localhost/$DB_NAME\""
