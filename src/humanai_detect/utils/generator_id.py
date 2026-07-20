"""sample_id onekinden hangi LLM'in (uretici) veriyi urettigini cikaran yardimci.

groups.parquet/fused.parquet hicbir zaman uretici bilgisini ayri bir kolonda
tutmadi (sadece dokuman/prompt group_id'si) -- bu yuzden uretici-bazli
degerlendirme/egitim gerektiren her yerde ayni cikarim mantigi tekrar
yazilmasin diye tek bir yerde toplandi.
"""

from __future__ import annotations

GENERATOR_NAMES = ["human", "qwen", "gpt4o_mini", "claude_sonnet5"]


def infer_generator(sample_id: str) -> str:
    """sample_id'den ureticiyi dondurur: human | qwen | gpt4o_mini | claude_sonnet5.

    Kapsam: hem ai_raw_* hem de ai_humanized_backtranslate_ai_raw_* (back-translation
    orijinal ai_raw'in ureticisini miras alir, humanize adiminin kendisi bir LLM
    uretimi degil). Bilinmeyen bir onekle karsilasilirsa ValueError firlatir --
    sessizce "unknown" donup yanlislikla kirilim disinda birakilmasindansa erken
    patlamak tercih edildi (yeni bir uretici eklenirse burasi da guncellenmeli).
    """
    if sample_id.startswith("human"):
        return "human"
    if "_transformers_" in sample_id:
        return "qwen"
    if "_openai_" in sample_id:
        return "gpt4o_mini"
    if "_anthropic_" in sample_id:
        return "claude_sonnet5"
    raise ValueError(
        f"sample_id icin uretici cikarilamadi: {sample_id!r} "
        f"(bilinen onekler: human*, *_transformers_*, *_openai_*, *_anthropic_*)"
    )
