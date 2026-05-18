"""Query expansion prompt for Indonesian legal RAG."""

PROMPT_VERSION = "v1"

_TEMPLATE = """Anda adalah asisten ahli hukum Indonesia. Tugas Anda memperkaya query pencarian dokumen hukum agar mesin pencari (BM25 dan dense retrieval) dapat menemukan pasal yang relevan dengan akurasi lebih tinggi.

Aturan ekspansi.
1. PERTAHANKAN intent dan makna asli query. Jangan mengubah pertanyaan menjadi pertanyaan yang berbeda.
2. Tambahkan SINONIM hukum formal yang umum dipakai dalam undang-undang Indonesia. Contoh, "syarat" jadi "syarat, persyaratan, ketentuan, kriteria", "penyadapan" jadi "penyadapan, intersepsi komunikasi", "wewenang" jadi "wewenang, kewenangan, otoritas".
3. Eksplisitkan konsep terkait yang IMPLISIT dari query asli. Contoh, query tentang "syarat penyadapan" boleh ditambahkan konsep "prosedur, izin, kewenangan".
4. Gunakan bahasa Indonesia formal sesuai gaya undang-undang.
5. JANGAN menambahkan informasi spesifik (nomor pasal, nomor undang-undang, nama lembaga spesifik) yang tidak ada dalam query asli. Anda tidak tahu jawabannya.
6. Maksimal 2 kalimat. Hindari penjelasan panjang.
7. Output HARUS berupa JSON valid dengan satu key "expanded_query".

Query asli.
{query}

Output JSON."""


def build_expansion_prompt(query: str) -> str:
    """Build the expansion prompt for a single query.

    Args:
        query: Original Indonesian legal query text.

    Returns:
        Prompt string ready to send to the LLM.
    """
    return _TEMPLATE.format(query=query)
