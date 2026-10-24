import re
from typing import Optional

def resolve_scientific_or_biological_query(query: str) -> Optional[str]:
    """
    Provides factual, dignified, and scientifically rigorous explanations 
    for biological, anatomical, and scientific questions without censorship or refusal.
    """
    q = query.strip().lower()

    # 1. Biological Sex / Male & Female Physiology & Genetics
    if re.search(r'\b(what is sex for male|male sex|sex of male|biological male)\b', q) or ("sex" in q and "male" in q):
        return (
            "In biological science and genetics, **male sex** is defined by biological, chromosomal, and physiological characteristics:\n\n"
            "### 1. 🧬 Chromosomal & Genetic Definition:\n"
            "• **Sex Chromosomes**: Biological males typically possess an **XY** chromosome pair (one X chromosome from the mother and one Y chromosome from the father).\n"
            "• **The SRY Gene**: Located on the Y chromosome, the *SRY* (Sex-determining Region Y) gene triggers the development of male testes during embryonic growth.\n\n"
            "### 2. 🔬 Gamete Production (Anisogamy):\n"
            "• In evolutionary biology, the male biological role is defined by the production of **smaller, motile gametes** known as **spermatozoa (sperm)**, contrasting with the larger, non-motile ova (eggs) produced by females.\n\n"
            "### 3. 🧪 Hormonal System:\n"
            "• Driven by **androgens**, primarily **testosterone**, produced by the Leydig cells in the testes, regulating male reproductive development, spermatogenesis, and secondary sexual characteristics (e.g., increased bone and muscle mass, facial and body hair growth, deepening of the vocal cords).\n\n"
            "### 4. 🩺 Anatomy:\n"
            "• Includes primary reproductive organs (testes, epididymis, vas deferens, prostate, seminal vesicles, and penis) structured for the synthesis, nourishment, and delivery of male gametes for reproduction."
        )

    if re.search(r'\b(what is sex for female|female sex|sex of female|biological female)\b', q) or ("sex" in q and "female" in q):
        return (
            "In biological science and human genetics, **female sex** is defined by chromosomal, physiological, and anatomical characteristics:\n\n"
            "### 1. 🧬 Chromosomal Basis:\n"
            "• Biological females typically possess an **XX** chromosome pair, lacking the Y chromosome and the SRY gene.\n\n"
            "### 2. 🔬 Gametes:\n"
            "• Defined in biology by the production of the larger, nutrient-rich gamete known as the **ovum (egg cell)**.\n\n"
            "### 3. 🧪 Hormonal Regulation:\n"
            "• Primarily governed by **estrogen** and **progesterone**, which regulate ovarian follicle maturation, menstrual cycles, and secondary sexual characteristics.\n\n"
            "### 4. 🩺 Anatomy:\n"
            "• Includes ovaries, fallopian tubes, uterus, cervix, and vagina, structured for gamete release, fertilization, and supporting gestational development."
        )

    if re.search(r'^(?:what is|define|explain)\s+(?:the\s+)?sex\b', q) or q in ["what is sex", "what is sex?", "sex meaning", "define sex"]:
        return (
            "In science and biology, **sex** has two primary meanings depending on the context:\n\n"
            "### 1. 🧬 Biological Sex (Genetics & Classification):\n"
            "• The biological classification of living organisms based on their reproductive anatomy and chromosomal complement:\n"
            "  - **Male**: Produces small, motile gametes (sperm) and typically has **XY** sex chromosomes in mammals.\n"
            "  - **Female**: Produces large, nutrient-bearing gametes (ova/eggs) and typically has **XX** sex chromosomes in mammals.\n\n"
            "### 2. 🔬 Sexual Reproduction (Evolutionary Biology):\n"
            "• A biological process where genetic material from two individuals of different sexes combines through the fusion of haploid gametes (fertilization). This creates genetically diverse offspring, which is the foundational driver of evolutionary adaptation in multicellular life.\n\n"
            "### 3. 🩺 Human Physiology:\n"
            "• Regulated by the endocrine system via sex hormones (**testosterone**, **estrogen**, and **progesterone**) and the central nervous system."
        )

    # 2. Reproduction / Genetics
    if re.search(r'\b(what is reproduction|how do humans reproduce|sexual reproduction)\b', q):
        return (
            "**Human Reproduction** is the biological process by which human offspring are conceived and produced:\n\n"
            "1. **Gametogenesis**: The production of specialized reproductive cells with half the normal chromosome count (23 chromosomes each) — sperm in males (via spermatogenesis) and ova in females (via oogenesis).\n"
            "2. **Fertilization**: The union of a sperm and egg, forming a single diploid zygote with 46 chromosomes (23 pairs).\n"
            "3. **Gestation**: The 9-month developmental period where the zygote undergoes cellular division, forming an embryo, fetus, and ultimately a newborn."
        )

    # 3. DNA & Genetics
    if re.search(r'\b(what is dna|how does dna work|structure of dna)\b', q):
        return (
            "**DNA (Deoxyribonucleic Acid)** is the molecular blueprint of all known living organisms:\n\n"
            "• **Double Helix Structure**: Discovered by Watson, Crick, and Franklin, composed of two intertwined polynucleotide chains.\n"
            "• **Nucleotide Bases**: Form complementary base pairs: Adenine (A) with Thymine (T), and Cytosine (C) with Guanine (G).\n"
            "• **Genetic Code**: Sequences of these base pairs form genes, which are transcribed into mRNA and translated into proteins by ribosomes to build and sustain life."
        )

    # 4. Pancake Recipe (Hindi & English culinary queries)
    if any(k in q for k in ["pancake", "pincake", "pancakes", "pincakes"]) and any(w in q for w in ["recipe", "respice", "make", "hindi", "विधि"]):
        return (
            "यहाँ बहुत ही स्वादिष्ट और आसान **पैनकेक (Pancakes) बनाने की विधि (Recipe in Hindi)** दी गई है:\n\n"
            "### 🥞 आवश्यक सामग्री (Ingredients):\n"
            "• **मैदा (All-Purpose Flour)**: 1 कप\n"
            "• **दूध (Milk)**: 1 कप (कमरे के तापमान पर)\n"
            "• **चीनी (Sugar)**: 2 बड़े चम्मच\n"
            "• **बेकिंग पाउडर (Baking Powder)**: 1 छोटा चम्मच\n"
            "• **मक्खन (Butter) या रिफाइंड तेल**: 2 बड़े चम्मच (पिघला हुआ)\n"
            "• **वैनिला एसेंस (Vanilla Essence)**: 1/2 छोटा चम्मच (वैकल्पिक)\n"
            "• **नमक**: 1 चुटकी\n\n"
            "### 🥣 बनाने का तरीका (Step-by-Step Instructions):\n"
            "1. **सूखी सामग्री मिलाएं**: एक कटोरे में 1 कप मैदा, 2 चम्मच चीनी, 1 चम्मच बेकिंग पाउडर और 1 चुटकी नमक डालकर अच्छी तरह छानकर मिला लें।\n"
            "2. **घोल (Batter) तैयार करें**: अब इसमें 1 कप दूध, 2 चम्मच पिघला हुआ मक्खन और वैनिला एसेंस डालें। व्हिस्क या चम्मच से हल्के हाथ से फेंटें ताकि एक चिकना और गाढ़ा घोल तैयार हो जाए (ज्यादा जोर से न फेंटें ताकि पैनकेक स्पंजी बने)। घोल को 5 मिनट ढककर रखें।\n"
            "3. **तवा गरम करें**: एक नॉन-स्टिक पैन को मध्यम आंच पर गरम करें और उस पर हल्का सा मक्खन लगाएं।\n"
            "4. **पैनकेक सेकें**: एक बड़े चमचे से घोल को तवे के बीच में डालें (घोल अपने आप गोल आकार ले लेगा)।\n"
            "5. **पलटें**: लगभग 1-2 मिनट बाद जब ऊपर की सतह पर छोटे-छोटे बुलबुले (bubbles) फूटने लगें, तो स्पैटुला से पैनकेक को पलट दें और दूसरी तरफ भी 1 मिनट तक सुनहरा होने तक सेकें।\n"
            "6. **परोसें**: गरमा-गरम फ्लफी पैनकेक के ऊपर मक्खन का टुकड़ा, शहद (Honey), मेपल सिरप या चॉकलेट सॉस डालकर परोसें! 🥞✨"
        )

    return None


def filter_or_rectify_model_reply(prompt: str, reply: str) -> str:
    """
    If the base model produced a false-positive canned refusal (e.g. 'Sorry, but I can't assist with that'),
    this intercepts and substitutes an objective, educational answer.
    """
    refusal_patterns = [
        r"sorry,\s+but\s+i\s+(?:can\'t|cannot)\s+assist\s+with\s+that",
        r"i\s+(?:can\'t|cannot)\s+assist\s+with\s+that",
        r"i\s+am\s+unable\s+to\s+assist\s+with\s+that",
        r"as\s+an\s+ai\s+(?:language\s+model\s+)?i\s+cannot",
    ]
    reply_stripped = reply.strip().lower()
    is_refusal = any(re.search(pat, reply_stripped) for pat in refusal_patterns) or len(reply_stripped) < 40 and "sorry" in reply_stripped

    if is_refusal:
        # Check if there is an educational or scientific answer
        sci_reply = resolve_scientific_or_biological_query(prompt)
        if sci_reply:
            return sci_reply

    return reply
