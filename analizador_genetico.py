from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, gc_fraction, IsoelectricPoint
import os, re, json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---- Funciones bioquímicas ----
def detectar_tipo(secuencia: str) -> str:
    secuencia = secuencia.strip().upper()
    if re.fullmatch(r"[ATCGN]+", secuencia):
        return "ADN"
    elif re.fullmatch(r"[AUCGN]+", secuencia):
        return "ARN"
    elif re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", secuencia):
        return "Proteína"
    return "Desconocido"


def analizar_secuencia(secuencia: str) -> dict:
    tipo = detectar_tipo(secuencia)
    secuencia = secuencia.upper().replace(" ", "").replace("\n", "")
    resultado = {"tipo": tipo, "entrada": secuencia}

    if tipo == "ADN":
        seq = Seq(secuencia)
        arn = seq.transcribe()
        prot = arn.translate(to_stop=True)
        resultado.update({
            "ARN": str(arn),
            "Proteína": str(prot),
            "GC%": round(gc_fraction(seq) * 100, 2),
            "Peso_molecular_DNA": round(molecular_weight(seq, "DNA"), 2)
        })
    elif tipo == "ARN":
        seq = Seq(secuencia)
        prot = seq.translate(to_stop=True)
        resultado.update({
            "Proteína": str(prot),
            "GC%": round(gc_fraction(seq.back_transcribe()) * 100, 2),
            "Peso_molecular_RNA": round(molecular_weight(seq, "RNA"), 2)
        })
    elif tipo == "Proteína":
        seq = Seq(secuencia)
        try:
            ip = IsoelectricPoint(seq)
            pI = round(ip.pi(), 3)
        except Exception:
            pI = "No calculable"
        resultado.update({
            "Peso_molecular_proteína": round(molecular_weight(seq, "protein"), 2),
            "Punto_isoelectrico": pI,
            "Longitud": len(secuencia)
        })
    else:
        resultado["error"] = "No se pudo identificar el tipo de secuencia."
    return resultado


# ---- Rutas Flask ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json()
    secuencia = data.get("secuencia", "")
    return jsonify(analizar_secuencia(secuencia))


@app.route("/upload_pdb", methods=["POST"])
def upload_pdb():
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({"filename": file.filename})


if __name__ == "__main__":
    app.run(debug=True)
