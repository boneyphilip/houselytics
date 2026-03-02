mkdir -p ~/.streamlit/

cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
port = ${PORT}
enableCORS = false
EOF