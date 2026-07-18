"""Tum stilometrik ozellik gruplarini tek sozlukte birlestirir.

Girdi  : ProcessedSample (Asama 2 ciktisi) + opsiyonel insan-korpusu referans istatistikleri
Cikti  : {feature_name: float} — Asama 4 embedding'i ile erken-fuzyon icin hazir

Referans istatistikleri (reference sozlugu):
    word_freqs         : dict[str, float]  — insan egitim kumesinin normalize edilmis kelime frekanslari
    mean_dep_depth     : float             — insan kumesinin ortalama dep. derinligi
    conjunction_density: float             — insan kumesinin ortalama baglac yogunlugu
Referans verilmezse (None), ilgili ozellikler NaN olarak dondurulur
(egitim sirasinda referans hesaplanip iletilecek).

lexicons sozlugu (opsiyonel, scripts/mine_contrastive_lexicons.py ciktisi):
    ai_cliche          : set[str]  — ai_raw korpusunda asiri temsil edilen n-gramlar
    human_informality  : set[str]  — insan korpusunda asiri temsil edilen n-gramlar
lexicons verilmezse ilgili yogunluk ozellikleri 0.0 doner (eslesecek n-gram yok).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humanai_detect.preprocessing.schemas import ProcessedSample

from . import discourse, lexical, lexicon, orthographic, readability, statistical, syntactic

# Asama 3 feature vektorunde yer alacak sabit POS etiket listesi
_POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX",
             "CCONJ", "SCONJ", "PART", "NUM", "PUNCT", "X"]


def _word_freqs(tokens: list[str]) -> dict[str, float]:
    """Token listesinden normalize edilmis kelime frekansi sozlugu uretir."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in counts.items()}


def extract_all_features(
    sample: "ProcessedSample",
    feature_config: dict,
    reference: dict | None = None,
    lexicons: dict | None = None,
) -> dict[str, float]:
    """Tek bir ProcessedSample icin aktif ozellikleri hesaplayip dondurur.

    reference verilmezse referans gerektiren ozellikler (kl_div, depth_shift,
    conjunction_deviation) NaN doner — training pipeline'i bunu doldurur.
    lexicons verilmezse n-gram yogunluk ozellikleri 0.0 doner.
    """
    feats: dict[str, float] = {}
    tokens = sample.tokens
    pos_tags = sample.pos_tags
    sentences = sample.sentences
    dep_parse = sample.dep_parse

    # --- Lexical ---
    lex_cfg = feature_config.get("lexical", {})
    if lex_cfg.get("type_token_ratio"):
        feats["ttr"] = lexical.type_token_ratio(tokens)
    if lex_cfg.get("hapax_legomena_ratio"):
        feats["hapax_ratio"] = lexical.hapax_legomena_ratio(tokens)
    if lex_cfg.get("mean_word_length"):
        feats["mean_word_len"] = lexical.mean_word_length(tokens)
    if lex_cfg.get("word_length_std"):
        feats["word_len_std"] = lexical.word_length_std(tokens)
    if lex_cfg.get("vocabulary_richness_yule_k"):
        feats["yule_k"] = lexical.vocabulary_richness_yule_k(tokens)

    # --- Statistical ---
    stat_cfg = feature_config.get("statistical", {})
    ngram_cfg = stat_cfg.get("ngram_entropy", {})
    if ngram_cfg.get("enabled"):
        for n in ngram_cfg.get("orders", [1, 2, 3]):
            feats[f"entropy_{n}gram"] = statistical.ngram_entropy(tokens, n)
    if stat_cfg.get("kl_divergence_word_freq") and reference and "word_freqs" in reference:
        feats["kl_div_word_freq"] = statistical.kl_divergence_word_freq(
            _word_freqs(tokens), reference["word_freqs"]
        )
    else:
        feats["kl_div_word_freq"] = math.nan
    if stat_cfg.get("perplexity"):
        feats["perplexity"] = sample.perplexity
    if stat_cfg.get("perplexity_ratio"):
        # log-oran kullanilir: ham oran (p1/p2) asiri carpik/heavy-tailed cikiyor (gozlemsel
        # aralik ~0.0005-4.17), bu da length_residualize.py'nin dogrusal regresyonuyla
        # uzunluk-confound'unu temizleyemiyordu (residual sonrasi r=0.65, HAM orandan (-0.46)
        # bile kotu). log(oran) = log(p1)-log(p2) ayni bilgiyi tasir (Spearman monotonic-
        # invariant, siralama degismez) ama residualizasyonun dogrusal varsayimina cok daha
        # uygun bir olcekte -- ayni teknikle residual sonrasi r=0.10'a duser (bkz. proje notlari).
        feats["perplexity_ratio"] = math.log(sample.perplexity_ratio) if sample.perplexity_ratio > 0 else 0.0
    if stat_cfg.get("burstiness_index"):
        feats["burstiness"] = sample.burstiness

    # --- Token-rank (GLTR-tarzi, causal LM ile preprocess asamasinda hesaplandi) ---
    rank_cfg = feature_config.get("token_rank", {})
    if rank_cfg.get("mean_rank"):
        feats["mean_token_rank"] = sample.mean_token_rank
    if rank_cfg.get("frac_rank_top1"):
        feats["frac_rank_top1"] = sample.frac_rank_top1
    if rank_cfg.get("frac_rank_top5"):
        feats["frac_rank_top5"] = sample.frac_rank_top5
    if rank_cfg.get("frac_rank_top10"):
        feats["frac_rank_top10"] = sample.frac_rank_top10
    if rank_cfg.get("rank_entropy"):
        feats["rank_entropy"] = sample.rank_entropy

    # --- Sentence-level (istatistiksel alt kume) ---
    sent_cfg = feature_config.get("sentence_level", {})
    sent_word_counts = [len(s.split()) for s in sentences]
    if sent_cfg.get("sentence_length_mean"):
        feats["sent_len_mean"] = statistics.mean(sent_word_counts) if sent_word_counts else 0.0
    if sent_cfg.get("sentence_length_variance"):
        feats["sent_len_var"] = statistics.variance(sent_word_counts) if len(sent_word_counts) > 1 else 0.0

    # --- Syntactic ---
    syn_cfg = feature_config.get("syntactic", {})
    if syn_cfg.get("mean_dependency_depth"):
        feats["mean_dep_depth"] = syntactic.mean_dependency_depth(dep_parse)
    if syn_cfg.get("dependency_depth_std"):
        feats["dep_depth_std"] = syntactic.dependency_depth_std(dep_parse)
    if syn_cfg.get("pos_distribution"):
        pos_dist = syntactic.pos_distribution(pos_tags)
        for tag in _POS_TAGS:
            feats[f"pos_{tag.lower()}"] = pos_dist.get(tag, 0.0)
    if syn_cfg.get("syntactic_depth_shift"):
        if reference and "mean_dep_depth" in reference:
            candidate_depths = syntactic._word_depths(dep_parse)
            feats["syntactic_depth_shift"] = syntactic.syntactic_depth_shift(
                [reference["mean_dep_depth"]], [statistics.mean(candidate_depths)] if candidate_depths else [0.0]
            )
        else:
            feats["syntactic_depth_shift"] = math.nan

    # --- Discourse ---
    dis_cfg = feature_config.get("discourse", {})
    conj_dens = discourse.conjunction_density(tokens, pos_tags)
    if dis_cfg.get("conjunction_density"):
        feats["conjunction_density"] = conj_dens
    if dis_cfg.get("conjunction_deviation_index"):
        ref_conj = reference.get("conjunction_density", 0.0) if reference else None
        feats["conjunction_deviation"] = (
            discourse.conjunction_deviation_index(conj_dens, ref_conj)
            if ref_conj is not None
            else math.nan
        )
    if dis_cfg.get("function_word_ratio"):
        feats["function_word_ratio"] = discourse.function_word_ratio(tokens, pos_tags)
    if dis_cfg.get("lexical_coherence"):
        feats["lexical_coherence"] = discourse.lexical_coherence(sentences)

    # --- Lexicon (contrastive-mining n-gram yogunlugu) ---
    lex_dens_cfg = feature_config.get("lexicon", {})
    if lex_dens_cfg.get("ai_cliche_density"):
        ai_lexicon = lexicons.get("ai_cliche", set()) if lexicons else set()
        feats["ai_cliche_density"] = lexicon.ngram_density(tokens, ai_lexicon)
    if lex_dens_cfg.get("human_informality_density"):
        human_lexicon = lexicons.get("human_informality", set()) if lexicons else set()
        feats["human_informality_density"] = lexicon.ngram_density(tokens, human_lexicon)

    # --- Orthographic (regex-tabanli, model/lexicon gerektirmez) ---
    ortho_cfg = feature_config.get("orthographic", {})
    if ortho_cfg.get("punct_irregularity_rate"):
        feats["punct_irregularity_rate"] = orthographic.punct_irregularity_rate(sample.cleaned_text)
    if ortho_cfg.get("double_space_rate"):
        feats["double_space_rate"] = orthographic.double_space_rate(sample.text)
    if ortho_cfg.get("post_punct_case_irregularity_rate"):
        feats["post_punct_case_irregularity_rate"] = orthographic.post_punct_case_irregularity_rate(sample.cleaned_text)

    # --- Readability ---
    read_cfg = feature_config.get("readability", {})
    if read_cfg.get("atesman"):
        feats["atesman"] = readability.atesman_score(sentences, tokens)
    if read_cfg.get("bezirci_yilmaz"):
        feats["bezirci_yilmaz"] = readability.bezirci_yilmaz_score(sentences, tokens)

    return feats
