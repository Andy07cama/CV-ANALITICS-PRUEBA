from sentence_transformers import SentenceTransformer, util

# Cargar el modelo de comparación
modelo = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Texto del CV
cv_texto = """
Soy estudiante de programación con experiencia en desarrollo web,
manejo HTML, CSS y JavaScript. También trabajé en equipo en proyectos escolares.
"""

# Texto de los requisitos del trabajo
requisitos_texto = """
Buscamos desarrollador web con conocimientos en HTML, CSS y trabajo en equipo.
"""

# Convertir los textos a embeddings
cv_embed = modelo.encode(cv_texto, convert_to_tensor=True)
req_embed = modelo.encode(requisitos_texto, convert_to_tensor=True)

# Calcular la similitud entre los textos
similitud = util.pytorch_cos_sim(cv_embed, req_embed)
porcentaje = similitud.item() * 100

# Mostrar resultados
print(f"Similitud entre CV y requisitos: {porcentaje:.2f}%")

# Feedback automático
if porcentaje > 75:
    print("🟢 ¡Tenés altas chances de ser aceptado para este trabajo!")
elif porcentaje > 50:
    print("🟡 Cumplís con algunos requisitos, pero podrías mejorar tu CV.")
else:
    print("🔴 Te faltan varios requisitos. Intentá reforzar tu CV.")
