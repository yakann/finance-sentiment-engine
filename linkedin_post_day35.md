# LinkedIn Post — Day 35

---

**AI projeleri test edilmeden ship edilmemeli.**

35 günlük bir AI mühendisliği öğrenme sürecinde en kritik dersi son haftada öğrendim: **ne kadar ölçerseniz, o kadar anlarsınız.**

Bu haftanın odağı, geliştirdiğim finance-sentiment-engine'i kapsamlı biçimde değerlendirmekti. İşte yaptıklarım:

---

**Ne yaptım?**

50 etiketli finans haberi içeren bir golden dataset oluşturdum. Ironic bullish haberler ("CEO ayrılıyor ama hisse uçuyor"), mixed signals ("rekor gelir açıkladı ama rehber düşük"), sell-the-news vakaları — gerçek dünyada karşılaşılan senaryolar.

Bu dataset'i 6 farklı dil modeline karşı koşturdum:
- OpenAI gpt-4.1-mini ve gpt-4.1-nano
- Groq üzerinde Meta Llama 3.3-70B ve 3.1-8B
- Anthropic Claude Haiku (yeni ve eski versiyon)

3 farklı RAG implementasyonunu (numpy cosine, Qdrant, LangChain + Cohere rerank) 20 soruyla NVIDIA 10-K'ya karşı ölçtüm. LangSmith ve Braintrust platformlarını karşılaştırdım.

---

**Şaşırtan bulgu:**

Claude Haiku 4.5, 6 model içinde en yüksek doğruluğa ulaştı. (%88 sentiment accuracy, 0.86 reasoning quality)

Claude Sonnet ile kıyaslandığında: **Sonnet'in ~%96'sı kadar performans, ~%18 maliyetiyle.**

Yapılandırılmış sınıflandırma görevlerinde "daha büyük model = daha iyi" varsayımı her zaman doğru değil. Haiku, kesin çıktı şeması (Pydantic) verildiğinde Sonnet'e rakip oluyor.

RAG tarafında da ilginç bir sonuç: Cohere rerank eklenmesi, sadece bu tek değişkenle faithfulness skorunu %5.4 artırdı. Retrieval kalitesi, cevap kalitesinin önündeki en kritik engel.

---

**Neden önemli?**

Çünkü bu bulgular olmadan yanlış seçim yapardım:
- Production'da pahalı bir model kullanmaya devam ederdim (gereksiz yere)
- RAG pipeline'ımda reranking olmadan devam ederdim (kalite kaybıyla)
- "Çalışıyor görünüyor" ile "ölçülmüş çalışıyor" arasındaki farkı göremezdim

Eval altyapısı bir maliyet değil, bir sigorta. Ve zaman zaman sürpriz gelir karşınıza.

---

35 günlük bu yolculukta scraper'dan başlayarak multi-agent LangGraph sistemine, oradan da kapsamlı eval pipeline'ına ulaştım. Her gün bir kavram, her gün bir commit.

Kaynak kodu ve tüm değerlendirme raporları GitHub'da:
👉 github.com/[username]/finance-sentiment-engine

**Siz AI projelerinizde evaluasyon için hangi araçları kullanıyorsunuz?**

#AIEngineering #LLM #MachineLearning #Python #LangChain #OpenAI #Anthropic

---

*Not: [username] kısmını kendi GitHub kullanıcı adınızla değiştirin.*
