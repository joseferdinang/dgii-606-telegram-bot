# Directorio de Credenciales

Si vas a utilizar **Google Cloud Vision OCR**, coloca tu archivo JSON de cuenta de servicio aquí con el nombre:
`google-vision.json`

En tu archivo `.env`, descomenta la variable:
```env
GOOGLE_VISION_CREDENTIALS=/app/credentials/google-vision.json
```

Si vas a utilizar únicamente **OpenAI ChatGPT** (`OPENAI_API_KEY`), no es necesario colocar ningún archivo aquí.
