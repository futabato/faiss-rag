# AIエンジニアリング実践講座Day3任意宿題

## 1. 独自の質問と参照資料の作成
- 自分で5つ以上の質問文を考案してください
- 各質問に対する回答を含む参照文書を用意してください
- 少なくとも1つは、LLMが単体では正確に答えられないような知識を含む質問にしてください

### 質問

- この仕様書に書かれている、エラーコード: 108のときの「発生した場合の対処法」を教えてください。
- この仕様書に書かれている、エラーコード: 110のときの「発生した場合の対処法」を教えてください。
- この仕様書に書かれている、エラーコード: 3E5のときの「発生した場合の対処法」を教えてください。
- この仕様書に書かれている、エラーコード: 4F2のときの「発生した場合の対処法」を教えてください。
- 取引履歴の参照・検索について教えてください。

## 2. 実験の実施
- 演習で使用したコードをベースに、以下の2つの方法で回答を生成してください
    - ベースのLLM(RAGなし)での回答生成
    - RAGを組み合わせた回答生成
- 回答の評価では、単純なYes/No判定でも良いです
    - より詳細な評価指標も検討していただけるとなお良いです

上記質問リストに対する回答が以下のものです。  
評価は作者の想定通りならYes、想定通りでなければNoです。

| 質問 | 回答 | 評価 |
| :-: | :-: | :-: |
| この仕様書に書かれている、エラーコード: 108のときの「発生した場合の対処法」を教えてください。 | 回答: 通信環境をご確認の上、もう一度お試しください。 | Yes |
| この仕様書に書かれている、エラーコード: 110のときの「発生した場合の対処法」を教えてください。 | この文書には記載されていません。 | Yes |
| この仕様書に書かれている、エラーコード: 3E5のときの「発生した場合の対処法」を教えてください。 | 回答: カスタマーサポートへお問い合わせください。 | Yes |
| この仕様書に書かれている、エラーコード: 4F2のときの「発生した場合の対処法」を教えてください。 | この文書には記載されていません。| Yes |
| 取引履歴の参照・検索について教えてください。 | 回答: エラーコード: 1001 内容: 取引履歴が見つかりません。対処法: 入力した条件に該当する取引履歴が存在しない場合、再度条件を確認して検索してください。 | No |

## 3. 結果分析と考察
- 生成した結果をまとめ、RAGありとRAGなしの差異を分析してください
- RAGによって回答が改善したケースと悪化したケースの両方について考察してください
- 結果に基づいて、RAGの有効性と限界についての考察を記述してください

結果分析と考察

- エラーコードをKeyとする内容の取得はうまく機能しています。
- 基本的に仕様書内の内容を尋ねているため、RAGなしでは見当違いの回答が返されます。
- うまくいかなかった例の理由は、チャンクの粒度の問題もありつつ、ページタイトル部分と本文部分と結びつけができて検索可能になるものであり、FAQに特化させたSystemPrompt故にハルシネーションが発生していると見られます。

## 質問設計の観点と意図

- エラーコードを含む自然言語で検索をして、回答を生成するようなRAGです。FAQマニュアルを想定して、インターネットから取得しました。
- 質問は、エラーコードに対応できるかどうかを第一の観点として設計しています。

## RAGの実装方法と工夫点

- LangChainとFaissと使ってシンプルなRAGを構築しました。拡張性よりも、クイックに実現可能であるところを有線して、LangChainを採用しています。
- SystemPromptは丁寧に与えています。FAQ部分は確実に取得できるように改良を重ねました。
- 辞書とベクトルDBを採用することにより、エラーコードのようなベクトルDBの検索で引っ掛かりずらいものを確実に取得します。
    - RAGを構築する中で、エラーコードはベクトル化する段階であまり意味のない文字列として解釈され、検索で引っ掛かりづらいということがわかりました。
    - Faissにおける `filter` は、ハッシュテーブルではなく、また検索対象を検索後にPythonレベルでフィルタリングする仕組みなのでO(N)かかりますが、辞書によりO(1)で確実に取得できます。

## 結果の分析と考察
