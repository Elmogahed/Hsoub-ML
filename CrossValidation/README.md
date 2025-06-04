#  iPhone Purchase Prediction – Classification Model Comparison

مشروع يهدف إلى مقارنة عدة نماذج من خوارزميات تعلم الآلة لتوقع قرار الشراء بناءً على خصائص العملاء، باستخدام بيانات ملف `iPhone.csv`.

---

##  الهدف من المشروع

- توقع ما إذا كان العميل سيشتري هاتف iPhone أم لا.
- مقارنة دقة نماذج التصنيف التالية:
  - Decision Tree
  - Gaussian Naive Bayes
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
- استخدام التحقق المتقاطع (Cross Validation) للحصول على تقييم أكثر دقة.

---

##  خطوات العمل

1. **قراءة البيانات:** تحميل بيانات العملاء من ملف `iPhone.csv`.
2. **التحويل والترميز:** ترميز البيانات النصية مثل الجنس باستخدام `LabelEncoder`.
3. **التقييس:** تطبيق `StandardScaler` لتوحيد البيانات.
4. **التقييم:** استخدام `cross_val_score` مع `accuracy` لمقارنة أداء النماذج.
5. **النتائج:** عرض نتائج الدقة لكل نموذج في جدول مقارن.

---

##  المتطلبات

```bash
pandas
numpy
scikit-learn
