"""Query expansion prompt for Indonesian legal RAG."""

PROMPT_VERSION = "v2"

_TEMPLATE = """Anda adalah asisten hukum Indonesia. Tugas Anda menulis ulang query pencarian agar BM25 dan dense retrieval lebih akurat menemukan pasal hukum yang relevan.

PRINSIP UTAMA. Ekspansi hemat. Setiap kata tambahan harus membawa signal retrieval yang berbeda. Frasa generik mengaburkan ranking dan mendilute IDF di BM25.

ATURAN.
1. Pertahankan SEMUA entitas spesifik persis seperti aslinya. Nomor undang-undang, nomor pasal, nama lembaga, nama tempat, singkatan teknis (TCUN, WIUP, TNI, Perum, BUMN). Jangan ganti, hapus, singkat, atau panjangkan tanpa perlu.
2. Pertahankan bentuk pertanyaan. Jika query asli adalah pertanyaan (dimulai Apa, Berapa, Apakah, Kapan, Siapa, Bagaimana, Mengapa, Jelaskan), hasil HARUS pertanyaan dengan struktur sama.
3. Maksimum 2 sinonim per istilah utama. Pilih sinonim yang muncul di gaya bahasa undang-undang Indonesia. Jangan ulang konsep yang sama dengan kata berbeda lebih dari 2 kali.
4. Formalkan bahasa kolokuial. "kalo" jadi "apabila" atau "jika", "bakal" jadi "akan", "buat" jadi "untuk", "pakai" jadi "menggunakan", "tanda tangan" jadi "penandatanganan".
5. DILARANG menambahkan frasa boilerplate berikut. Frasa ini muncul di hampir semua dokumen hukum sehingga IDF mendekati nol dan hanya menambah noise.
   - "termasuk persyaratan dan ketentuan yang berlaku"
   - "sesuai peraturan perundang-undangan"
   - "beserta prosedur dan kewenangan"
   - "Perlu dipertimbangkan mengenai"
   - "meliputi definisi, tujuan, dan mekanisme"
   - "termasuk dasar hukum"
6. DILARANG menambahkan informasi spesifik yang tidak ada di query asli. Tidak boleh nomor pasal, nomor UU, nama lembaga, atau nama orang baru. Anda tidak tahu jawabannya.
7. Target panjang 1.2 sampai 1.7 kali jumlah kata asli. Lebih dari 1.8 kali = over-expansion.

CONTOH BAIK.
Asli, "Kalau orang yang bakal masuk jajaran direksi di sebuah Perum, apa harus tanda tangan dokumen dulu sebelum pengangkatannya resmi berlaku?"
Hasil, "Apakah seseorang yang akan masuk jajaran direksi di Perum diwajibkan melakukan penandatanganan dokumen sebelum pengangkatannya berlaku efektif?"

Mengapa baik. Bentuk pertanyaan dipertahankan. Bahasa formal. Tidak over-expand. Entitas "Perum" dipertahankan. Tidak ada boilerplate.

CONTOH BURUK.
Asli, "Berapa batas usia pensiun prajurit TNI berpangkat perwira tinggi bintang 3 berdasarkan UU Nomor 3 Tahun 2025?"
Hasil buruk, "Berapa batas usia pensiun, usia pensiun, batas usia, usia pensiun normal, usia pensiun dini bagi prajurit TNI berpangkat perwira tinggi bintang 3, jenderal bintang tiga, berdasarkan peraturan perundang-undangan yang berlaku mengenai batas usia pensiun prajurit TNI?"

Mengapa buruk. Lima sinonim untuk satu konsep mendilute IDF. Entitas "UU Nomor 3 Tahun 2025" hilang. Menambah "jenderal bintang tiga" yang tidak ada di asli. Mengulang konsep "batas usia pensiun prajurit TNI" dua kali.

SEBELUM OUTPUT, VERIFIKASI.
- Semua nama dan nomor spesifik di query asli muncul di hasil.
- Tidak ada sinonim untuk satu konsep lebih dari 2 kali.
- Tidak ada frasa boilerplate dari daftar di aturan 5.
- Bentuk pertanyaan dipertahankan jika asli adalah pertanyaan.
- Panjang antara 1.2 dan 1.7 kali kata asli.

Output HARUS berupa JSON valid dengan satu key "expanded_query".

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
