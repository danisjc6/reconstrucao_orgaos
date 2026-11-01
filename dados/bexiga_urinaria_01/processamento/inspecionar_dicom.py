
from pathlib import Path
import pydicom

from pathlib import Path

pasta = (
    Path.home()
    / "Documentos"
    / "reconstrucao_orgaos"
    / "dados"
    / "bexiga_urinaria_01"
    / "configuracao"
    / "dicom"
)

arquivos = sorted(
    p for p in pasta.iterdir()
    if p.is_file() and p.suffix.lower() == ".dcm"
)

print(f"Total de arquivos DICOM: {len(arquivos)}")

if not arquivos:
    raise SystemExit("Nenhum arquivo DICOM encontrado.")

for arquivo in arquivos[:3]:
    ds = pydicom.dcmread(arquivo, stop_before_pixels=True)

    print(f"\nArquivo: {arquivo.name}")
    print("Dimensões:", ds.get("Rows"), "x", ds.get("Columns"))
    print("PixelSpacing:", ds.get("PixelSpacing", "Não informado"))
    print("ImagerPixelSpacing:", ds.get("ImagerPixelSpacing", "Não informado"))
    print("PhotometricInterpretation:",
          ds.get("PhotometricInterpretation", "Não informado"))
    print("BitsStored:", ds.get("BitsStored", "Não informado"))
    print("RescaleSlope:", ds.get("RescaleSlope", "Não informado"))
    print("RescaleIntercept:", ds.get("RescaleIntercept", "Não informado"))
