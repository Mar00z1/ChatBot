import os
import openai
import numpy as np
import requests
import time

# Configura tu clave API de OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY", "tu-api-key-aqui")

def fetch_remoteok_jobs(max_results: int = 20) -> list:
    """Obtiene trabajos remotos con manejo de rate limiting"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        print("Obteniendo trabajos (respeta el rate limiting)...")
        response = requests.get(
            "https://remoteok.io/api?tags=dev",
            headers=headers,
            timeout=15
        )
        
        # Verificar rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limit alcanzado. Reintentando en {retry_after} segundos...")
            time.sleep(retry_after)
            return fetch_remoteok_jobs(max_results)  # Reintento recursivo
            
        response.raise_for_status()
        
        jobs_data = response.json()[1:]
        return [process_job(job) for job in jobs_data[:max_results]]
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def process_job(job):
    """Procesa individualmente cada trabajo"""
    time.sleep(1)  # Delay entre procesamientos
    return {
        "id": job.get("id"),
        "title": job.get("position"),
        "company": job.get("company"),
        "skills": job.get("tags", []),
        "description": f"{job.get('description', '')}",
        "url": job.get("url"),
        }
    

def get_embedding(text: str) -> list:
    """Obtiene el embedding del texto usando OpenAI"""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    """Calcula la similitud coseno entre dos vectores"""
    a_np = np.array(a)
    b_np = np.array(b)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

def main():
    print("=== Matching de Empleos para Estudiantes y Jóvenes Profesionales ===")
    print("Completa esta breve encuesta para encontrar oportunidades adecuadas:\n")
    
    # Nueva encuesta estructurada
    print("1. Carrera estudiada (ej: Ingeniería Informática, Administración):")
    carrera = input("   > ").strip()
    
    print("\n2. Experiencia Laboral (incluye prácticas/pasantías):")
    print("   Ejemplo: '6 meses en desarrollo web, 3 meses en análisis de datos'")
    experiencia = input("   > ").strip()
    
    print("\n3. Áreas de interés (ej: IA, Marketing Digital, Finanzas):")
    print("   Separa por comas los temas que te interesan")
    areas = [a.strip() for a in input("   > ").split(",") if a.strip()]
    
    print("\n4. Habilidades técnicas y blandas (ej: Python, Trabajo en equipo):")
    print("   Separa por comas lo que mejor sabes hacer")
    habilidades = [h.strip() for h in input("   > ").split(",") if h.strip()]
    
    # Crear perfil estructurado
    profile_text = (
        f"Carrera: {carrera}\n"
        f"Experiencia relevante: {experiencia}\n"
        f"Habilidades técnicas: {', '.join(habilidades)}\n"
        f"Áreas de interés profesional: {', '.join(areas)}"
    )

     # Obtener embedding del perfil
    profile_embedding = get_embedding(profile_text)

    print("\nBuscando trabajos remotos en RemoteOK...")
    jobs_db = fetch_remoteok_jobs()
    
    if not jobs_db:
        print("No se encontraron ofertas remotas.")
        return
    
    print(f"\nAnalizando {len(jobs_db)} trabajos remotos...")
    
    # Nuevo sistema de procesamiento de trabajos
    matches = []
    for job in jobs_db:
        # Mejorar el texto del puesto
        job_text = (
            f"Puesto: {job['title']}\n"
            f"Requisitos clave: {job['description']}\n"
            f"Habilidades requeridas: {', '.join(job['skills'])}\n"
            f"Sector: {' '.join(job['skills'])}"
        )
        
        # Limitar la longitud del texto para enfoque
        job_text = ' '.join(job_text.split()[:300])  # Máximo 300 palabras
        
        job_embedding = get_embedding(job_text)
        similarity = cosine_similarity(profile_embedding, job_embedding)
        
        # Calcular coincidencia de habilidades
        skill_match = len(set(habilidades) & set(job['skills']))
        skill_boost = min(skill_match * 0.05, 0.15)  # Hasta 15% de bonus
        
        matches.append({
            "job": job,
            "similarity": round(similarity + skill_boost, 2),
            "skill_matches": skill_match
        })

    # Ordenar y seleccionar los 3 mejores
    sorted_matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)[:3]
    
    # Generar explicaciones
    for match in sorted_matches:
        prompt = (
            f"Como orientador profesional, explica cómo el perfil del candidato "
            f"se alinea con el puesto de {match['job']['title']} considerando:\n"
            f"- Coincidencia de habilidades técnicas ({match['skill_matches']} en común)\n"
            f"- Experiencia relevante\n"
            f"- Formación académica\n"
            f"Perfil del candidato:\n{profile_text}\n\n"
            f"Detalles del puesto:\n{job_text}\n\n"
            f"La respuesta debe ser concisa (máx 200 palabras) y destacar:\n"
            f"1. Puntos fuertes del candidato\n2. Posibles gaps\n3. Recomendaciones"
        )
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4500,
            temperature=0.7
        )
        match["explanation"] = response.choices[0].message.content
    
    # Mostrar resultados
    print("\n=== Mejores Coincidencias Remotas ===")
    for idx, match in enumerate(sorted_matches, 1):
        job = match["job"]
        print(f"\n{idx}. {job['title']} - {job['company']}")
        print(f"   Similitud: {match['similarity']}")
        print(f"   Enlace: {job['url']}")
        print("   Explicación:")
        explanation = match['explanation'].replace('\n', '\n   ')
        print(f"   {explanation}")
        print("-" * 80)

if __name__ == "__main__":
    main()