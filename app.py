from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
from Bio.Seq import Seq
from Bio import SeqIO, SeqUtils
import io

app = Flask(__name__)
CORS(app)

print("🚀 Cargando modelo genético (flan-t5-base)...")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
print("✅ Modelo cargado correctamente y listo para análisis genético.")


# ----------------------------------------------------------
# 🧬 RUTA PRINCIPAL: recibe texto o secuencias desde index.html
# ----------------------------------------------------------
@app.route('/preguntar', methods=['POST'])
def preguntar():
    try:
        data = request.get_json(force=True)
        pregunta = data.get("pregunta", "").strip()
    except Exception:
        return jsonify({"respuesta": "⚠️ Error al interpretar la solicitud JSON."}), 400

    if not pregunta:
        return jsonify({"respuesta": "⚠️ No se recibió ninguna pregunta genética."}), 400

    texto_upper = pregunta.upper().replace(" ", "")

    # 1️⃣ Comparación de secuencias
    if "COMPARA" in texto_upper:
        respuesta_comp = comparar_secuencias_en_texto(pregunta)
        if respuesta_comp:
            return jsonify({"respuesta": respuesta_comp})

    # 2️⃣ Si es secuencia pura (ADN, ARN o proteína)
    if es_secuencia(texto_upper):
        tipo = detectar_tipo_secuencia(texto_upper)
        if tipo == "ADN":
            return jsonify({"respuesta": analizar_secuencia_dna(texto_upper)})
        elif tipo == "ARN":
            return jsonify({"respuesta": analizar_secuencia_arn(texto_upper)})
        elif tipo == "PROTEINA":
            return jsonify({"respuesta": analizar_proteina(texto_upper)})

    # 3️⃣ Preguntas teóricas
    respuesta_bio = procesar_pregunta_genetica(pregunta)
    if respuesta_bio:
        return jsonify({"respuesta": respuesta_bio})

    # 4️⃣ IA avanzada para explicaciones generales
    input_text = f"Explica con lenguaje genético avanzado: {pregunta}"
    resultado = qa_pipeline(input_text, max_new_tokens=200)
    respuesta = resultado[0]["generated_text"]

    return jsonify({"respuesta": respuesta})


# ----------------------------------------------------------
# 📂 RUTA PARA CARGAR ARCHIVOS FASTA O GENBANK
# ----------------------------------------------------------
@app.route('/cargar-secuencia', methods=['POST'])
def cargar_secuencia():
    """Permite subir un archivo FASTA o GenBank desde el frontend."""
    if 'archivo' not in request.files:
        return jsonify({"respuesta": "⚠️ No se envió ningún archivo."}), 400

    archivo = request.files['archivo']
    nombre = archivo.filename.lower()

    try:
        if nombre.endswith(".fasta") or nombre.endswith(".fa"):
            contenido = archivo.read().decode()
            seq_record = SeqIO.read(io.StringIO(contenido), "fasta")
        elif nombre.endswith(".gb") or nombre.endswith(".gbk"):
            contenido = archivo.read().decode()
            seq_record = SeqIO.read(io.StringIO(contenido), "genbank")
        else:
            return jsonify({"respuesta": "❌ Formato no compatible. Usa .fasta o .gbk"}), 400

        secuencia = str(seq_record.seq)
        tipo = detectar_tipo_secuencia(secuencia)
        if tipo == "ADN":
            resultado = analizar_secuencia_dna(secuencia)
        elif tipo == "ARN":
            resultado = analizar_secuencia_arn(secuencia)
        else:
            resultado = analizar_proteina(secuencia)

        return jsonify({
            "respuesta": f"📄 Archivo leído correctamente ({tipo})\n\n{resultado}"
        })
    except Exception as e:
        return jsonify({"respuesta": f"⚠️ Error al leer el archivo: {e}"}), 500


# ----------------------------------------------------------
# 🧩 FUNCIONES BIOLÓGICAS
# ----------------------------------------------------------
def es_secuencia(texto):
    """Verifica si es ADN, ARN o proteína."""
    return all(ch.isalpha() for ch in texto)

def detectar_tipo_secuencia(seq):
    """Detecta automáticamente el tipo de secuencia."""
    if set(seq).issubset({"A", "T", "C", "G"}):
        return "ADN"
    elif set(seq).issubset({"A", "U", "C", "G"}):
        return "ARN"
    else:
        return "PROTEINA"

def analizar_secuencia_dna(seq):
    dna = Seq(seq)
    gc = round(SeqUtils.gc_fraction(dna) * 100, 2)
    transcripcion = str(dna.transcribe())
    traduccion = str(dna.translate(to_stop=True))
    return (
        f"🧬 **Análisis de ADN**\n\n"
        f"📏 Longitud: {len(seq)} bases\n"
        f"🧪 GC%: {gc}\n"
        f"🔡 ARNm: {transcripcion[:80]}...\n"
        f"💠 Proteína: {traduccion[:50]}..."
    )

def analizar_secuencia_arn(seq):
    arnm = Seq(seq)
    traduccion = str(arnm.translate(to_stop=True))
    return (
        f"🧬 **Análisis de ARNm**\n\n"
        f"📏 Longitud: {len(seq)} bases\n"
        f"💠 Traducción: {traduccion[:50]}...\n"
        f"🧩 Observación: ARNm puede derivarse de ADN mediante transcripción inversa."
    )

def analizar_proteina(seq):
    aa_count = len(seq)
    peso = round(SeqUtils.molecular_weight(Seq(seq), seq_type="protein"), 2)
    return (
        f"🧫 **Análisis de Proteína**\n\n"
        f"🔹 Longitud: {aa_count} aminoácidos\n"
        f"⚖️ Peso molecular estimado: {peso} Da\n"
        f"💡 Interpretación: posible fragmento proteico funcional o dominio parcial."
    )


def procesar_pregunta_genetica(texto):
    texto_upper = texto.upper()
    if "CRISPR" in texto_upper:
        return "🧬 **CRISPR-Cas9**: sistema de edición genética que usa ARN guía para dirigir Cas9 al ADN."
    if "MUTACION" in texto_upper:
        return "🔍 **Mutación Genética**: cambio en la secuencia de ADN que puede modificar la proteína resultante."
    if "ARN" in texto_upper:
        return "💡 **Tipos de ARN**: ARNm (mensajero), ARNt (transferencia) y ARNr (ribosomal)."
    if "FASTA" in texto_upper or "GENBANK" in texto_upper:
        return "📁 Puedes subir archivos FASTA o GenBank en la sección de carga de secuencias."
    return None


def comparar_secuencias_en_texto(texto):
    texto_upper = texto.upper().replace(" ", "")
    secuencias, actual = [], ""
    for ch in texto_upper:
        if ch in "ATCG":
            actual += ch
        else:
            if len(actual) > 5:
                secuencias.append(actual)
            actual = ""
    if len(actual) > 5:
        secuencias.append(actual)

    if len(secuencias) < 2:
        return "❌ No se detectaron dos secuencias válidas."

    return interpretar_mutaciones(secuencias[0], secuencias[1])


def interpretar_mutaciones(seq1, seq2):
    mutaciones = [(i + 1, seq1[i], seq2[i]) for i in range(min(len(seq1), len(seq2))) if seq1[i] != seq2[i]]
    similitud = round((1 - len(mutaciones) / max(len(seq1), len(seq2))) * 100, 2)
    tipo = "idénticas" if not mutaciones else "con diferencias"
    respuesta = (
        f"🧬 **Comparación Genética**\n"
        f"Similitud: {similitud}%\n"
        f"Estado: {tipo}\n"
    )
    if mutaciones:
        respuesta += "🔬 Mutaciones detectadas:\n"
        for pos, a, b in mutaciones:
            respuesta += f"• Pos {pos}: {a}→{b}\n"
    return respuesta
# ----------------------------------------------------------
# 🧬 BLOQUE COMPLEMENTARIO - Lectura avanzada de FASTA / GenBank
# ----------------------------------------------------------
from io import StringIO

@app.route('/upload', methods=['POST'])
def upload():
    """
    Nueva ruta que permite subir archivos genéticos (FASTA / GenBank)
    y obtener un análisis estructurado: tipo de secuencia, longitud,
    %GC, transcripción y traducción.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se ha enviado ningún archivo"}), 400
        
        archivo = request.files['file']
        contenido = archivo.read().decode('utf-8')

        # Detectar formato automáticamente
        if archivo.filename.endswith(('.fasta', '.fa')):
            formato = 'fasta'
        elif archivo.filename.endswith(('.gb', '.gbk')):
            formato = 'genbank'
        else:
            return jsonify({"error": "Formato de archivo no compatible. Usa .fasta o .gb"}), 400

        # Leer con BioPython
        registros = list(SeqIO.parse(StringIO(contenido), formato))
        if not registros:
            return jsonify({"error": "No se encontró ninguna secuencia válida"}), 400

        resultado_total = []
        for record in registros:
            secuencia = str(record.seq).upper()

            # Detectar tipo de secuencia
            tipo = (
                "ADN" if set(secuencia) <= set("ATGC") else
                "ARN" if set(secuencia) <= set("AUGC") else
                "Proteína"
            )

            # Calcular GC%
            gc = round((secuencia.count("G") + secuencia.count("C")) / len(secuencia) * 100, 2) if len(secuencia) > 0 else 0

            analisis = {
                "ID": record.id,
                "Descripción": record.description,
                "Tipo de secuencia": tipo,
                "Longitud": len(secuencia),
                "Contenido GC (%)": gc
            }

            # Transcripción y traducción si aplica
            if tipo == "ADN":
                arn = secuencia.replace("T", "U")
                analisis["Transcripción (ARNm)"] = arn
                try:
                    analisis["Traducción (Proteína)"] = str(record.seq.translate(to_stop=True))
                except Exception:
                    analisis["Traducción (Proteína)"] = "No se pudo traducir"
            elif tipo == "ARN":
                adn = secuencia.replace("U", "T")
                analisis["Retrotranscripción (ADN)"] = adn

            resultado_total.append(analisis)

        return jsonify({
            "mensaje": f"Archivo procesado correctamente ({formato.upper()})",
            "datos": resultado_total
        })

    except Exception as e:
        return jsonify({"error": f"Error interno al analizar archivo: {str(e)}"}), 500


# ----------------------------------------------------------
# 🚀 EJECUCIÓN
# ----------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
