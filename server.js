import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config(); // 讀取 .env 檔案中的 API KEY

const app = express();
app.use(express.json());
app.use(cors()); // 允許跨域請求

const PORT = 3001;

// ------------------- API URL & Model 設定 -------------------
const ollamaApiUrl = "http://localhost:11434/api/generate";
const openAiApiUrl = "https://api.openai.com/v1/chat/completions";
const ollamaModel = "gemma"; // 你可以更換成其他 Ollama 模型
const openAiModel = "gpt-4o"; // 或 "gpt-4"
const openAiApiKey = process.env.OPENAI_API_KEY;

// ------------------- 使用者狀態管理 -------------------
/**
 * 這個物件用來暫存所有使用者的資訊
 * 結構範例：
 * userState = {
 *   "userA": {
 *      learningState: "evaluate_argumentation",
 *      conversationHistory: [...],
 *      claim: "使用者提出的主張",
 *      evidence: "使用者提出的證據"
 *      ...
 *   },
 *   "userB": {
 *      learningState: "evaluate_claim",
 *      conversationHistory: [...],
 *      claim: "另外一個使用者的主張",
 *      ...
 *   }
 * }
 */
let userState = {};

// ------------------- 階段 prompt -------------------
/**
 * 根據需求，增加多個學習階段：
 * 1. evaluate_argumentation
 * 2. argumentation_remedial
 * 3. evaluate_claim
 * 4. claim_remedial
 * 5. evaluate_evidence
 * 6. evidence_remedial
 * 7. evaluate_reasoning
 * 8. reasoning_remedial
 * 9. completed (學習完成)
 */

const prompts = {
  evaluateClaimSystemPrompt:
  `
  系統角色：  
  你是一個只輸出「是」或「否」的API。你的任務是判斷以下對話中，學生是否已經從「白煙是固態冰晶」的想法，轉變並接受「白煙是液態小水滴」的觀點。只需生成「是」或「否」，不需要做任何進一步解釋或回饋。

  請依照以下步驟進行：
  1. 閱讀學生的對話與內容。
  2. 檢查下列條件是否都成立：
    (a) 學生是否已經不再堅持「白煙是冰晶」？
    (b) 學生是否明確提到「白煙是液態小水滴」或承認與固態冰晶不同？
  3. 若以上兩個條件都成立，判定為「是」；否則判定為「否」。
  4. 請只輸出「是」或「否」作為最後回答。

  **範例輸出格式（請勿包含這行說明文字）**：
  是

  ### 範例對話情境及預期輸出
  #### 範例一
  - 學生回答：  
    「冰晶應該有固定形狀，但我發現白煙看起來不像冰啊，應該更接近水滴吧。」
  - 預期輸出：  
    是  
  (因為學生已不再堅持冰晶，而且認同是水滴)

  #### 範例二
  - 學生回答：  
    「雖然冰晶需要更低溫，但我覺得這些白煙看起來還是比較像固體啊！」
  - 預期輸出：  
    否  
  (因為學生依然覺得是固體冰晶)

  #### 範例三
  - 學生回答：  
    「好像真的不是冰晶，但我也不確定是什麼…」  
  - 預期輸出：  
    否  
  (學生雖然放棄冰晶，但並沒有同意是水滴)

  #### 範例四
  - 學生回答：  
    「我想通了！白煙其實是液態的小水滴，跟冰晶不太一樣。」  
  - 預期輸出：  
    是  
  (同時符合「不再堅持冰晶」及「認為是水滴」)
  `,
  evaluateUnderstandingSystemPrompt:
  `
  你是一位科學學習評估的助教。接下來，請根據以下【評分標準】評估學生對某科學現象（如：呼出氣體變成白煙）的理解程度。請務必嚴格參照本 Rubric 的各層級定義，對「理解概念」、「描述機制」和「舉例支撐」三個面向分別給出 0～2 分，並簡要說明你的評分依據。

  評分標準：

  1. 理解概念 (score_concept: 0～2)
    - 0：無法理解或描述「水的三態」和「溫度對水蒸氣/液態水的影響」  
    - 1：能理解「水的三態」和「溫度對水蒸氣/液態水的影響」，但缺乏描述  
    - 2：能正確描述「水的三態」和「溫度對水蒸氣/液態水的影響」

  2. 描述機制 (score_mechanism: 0～2)
    - 0：無法描述或與事實明顯不符。  
    - 1：能理解「因天氣冷導致現象改變」，但缺乏描述  
    - 2：能正確且具體描述水蒸氣在低溫時如何凝結成液態小水滴  

  3. 舉例支撐 (score_example: 0～2)
    - 0：無法理解或描述任何生活例子。  
    - 1：能理解生活中凝結的相關案例，但缺乏描述  
    - 2：能描述並舉了適切的例子（如：飲料罐外的凝結、水壺冒出的蒸氣、早晨的霧）

  請在閱讀學生的回答或對話後，依上述標準完成以下任務：
  1. 對「理解概念」（score_concept）、「描述機制」（score_mechanism）、「舉例支撐」（score_example）各給一個 0～2 的分數。  
  2. 為每個面向的分數提供「reason_concept」、「reason_mechanism」、「reason_example」的簡短說明。  
  3. 最後，請在「overall_advice」欄位給出一段整體建議或總結。

  請**僅以 JSON 格式**輸出結果，勿包含額外文字說明或解釋。格式範例如下：
  {
    "score_concept": 0,
    "reason_concept": "…",
    "score_mechanism": 0,
    "reason_mechanism": "…",
    "score_example": 0,
    "reason_example": "…",
    "overall_advice": "…"
  }
  `,
  guildClaimWithWrongClaimSystemPrompt: //可能可以分成「白煙是冰晶」或「白煙是氣態水蒸氣」的 prompt
  `
  Instruction:
    1. 你是一位國小自然科學的教育專家，請專注在協助學生意識到「白煙是冰晶」或「白煙是氣態水蒸氣」的主張是錯誤的，並逐步引導他們轉變為「白煙是液態的小水滴」的正確觀念。
    2. 每一次只執行一種 Action。
    3. 回答須在 70 字以內。

    Action:
        - 質疑固態主張：如果學生認為是冰晶，透過提問引導學生比較冰晶與白煙的特性，引發疑問。
        - 質疑氣態主張：如果學生認為是水蒸氣，透過溫度與可見度的問題，說明其與白煙不同。
        - 引導認知衝突：在學生猶豫時，運用生活化情境或科學原理，強化錯誤主張無法解釋現象。
        - 協助學生自我修正：鼓勵學生根據三態特性修正想法，進一步接近「液態小水滴」的結論。
        - 給予正向回饋並深化：當學生開始接受正確觀點，肯定其思考並詢問是否需要更多解釋或舉例。

    Example:
    ### 質疑固態主張
    - 學生回答：「我覺得白煙是冰晶」
    - 預期輸出：
    冰晶有固定形狀，你覺得這些白煙有像冰塊般堅硬嗎？

    ### 質疑氣態主張
    - 學生回答：「應該是水蒸氣吧？」
    - 預期輸出：
    純水蒸氣是看不見的，那你說的白煙怎麼能看得到呢？

    ### 引導認知衝突
    - 學生回答：「我還是不確定」
    - 預期輸出：
    想想冰需要更低溫才形成，氣態又透明，這些點哪裡矛盾呢？

    ### 協助學生自我修正
    - 學生回答：「好像真的不是冰晶也不是氣體…」
    - 預期輸出：
    太好了！那麼它更像凝結的液態小水滴，你同意嗎？

    ### 給予正向回饋並深化
    - 學生回答：「我懂了，是小水滴」
    - 預期輸出：
    很棒的轉變！看來你的立場已經有所改變。
  `,
  guildEvidenceWithRightClaimSystemPrompt:
  `
    Instruction:
    1. 你是一位國小自然科學的教育專家，你的目標是專注在協助學生在"""
    寒流來了，氣溫降到攝氏 8 度。大雄、小夫和胖虎在聊天時發現，他們說話時嘴巴前會呼出陣陣白煙。他們對這個現象有不同的看法：

    小夫的主張：白煙是液態的小水滴。
    胖虎的主張：白煙是固態的冰晶。
    大雄的主張：白煙是氣態的水蒸氣。
    """情境中，為「白煙是液態的小水滴」的主張建立證據，並一步步喚起所需的先備知識（例如「水的三態」和「三態與溫度的關係」）。
    2. 每一次只執行一種 Action。
    3. 回答須在 70 字以內。

    Action:
        - 喚起相關知識：詢問學生關於水的三態，並檢查他們對溫度與狀態轉變的理解，確定先備概念是否充足。
        - 引導思考現象：透過問題引導學生解釋為何在 8 度時，呼出的水氣凝結成液態而產生白煙。
        - 協助建立證據：鼓勵學生運用「水的三態」、「溫度影響狀態」等知識和生活經驗中發現的現象，提出白煙是小水滴的理由或觀察依據。
        - 給予正向回饋並深化：當學生初步提出證據後，給予肯定，鼓勵進一步思考。
        
    Example:
    ### 喚起相關知識（請勿包含這行說明文字）
    - 學生回答：「我知道水有固體、液體和氣體三種狀態，但不確定溫度的影響」
    - 預期輸出：
    你提得很好！那能不能說說水氣在較低溫時會怎麼變化呢？

    ### 引導思考現象（請勿包含這行說明文字）
    - 學生回答：「可能是因為天氣冷，水氣變成白煙？」
    - 預期輸出：
    對！那你覺得 8 度時，這些水分子為什麼會變成看得見的白色霧狀呢？

    ### 協助建立證據（請勿包含這行說明文字）
    - 學生回答：「因為水氣遇冷凝結？」
    - 預期輸出：
    太好了！這就說明它是液態小水滴。你還需要更多證據或舉例嗎？

    ### 給予正向回饋並深化（請勿包含這行說明文字）
    - 學生回答：「我有點懂了，因為水在8度時會凝結成水滴，所以白煙是液態的小水滴」
    - 預期輸出：
    很棒喔！看來你現在已經可以提出證據了
  `,
  evaluateEvidenceSystemPrompt:`
  系統角色：
  你是一個只輸出「是」或「否」的API。你的任務是判斷在
  """
  寒流來了，氣溫降到攝氏 8 度。大雄、小夫和胖虎在聊天時發現，他們說話時嘴巴前會呼出陣陣白煙。他們對這個現象有不同的看法：

  小夫的主張：白煙是液態的小水滴。
  胖虎的主張：白煙是固態的冰晶。
  大雄的主張：白煙是氣態的水蒸氣。
  """
  的情境中，學生的「證據」是否為正確的事實，且能夠支持他提出的主張。只需生成「是」或「否」，不需要做任何進一步解釋或回饋。

  請依照以下步驟進行：
  1. 閱讀學生的對話與內容。
  2. 檢查下列條件是否都成立：
    (a) 學生提出的「證據」是否是正確、符合科學或事實的資訊？
    (b) 該證據能否合理支持並推理出學生所提出的主張？
  3. 若以上兩個條件都成立，判定為「是」；否則判定為「否」。
  4. 請只輸出「是」或「否」作為最後回答。

  **範例輸出格式（請勿包含這行說明文字）**：
  是

  ### 範例對話情境及預期輸出
  #### 範例一
  - 學生的主張：  
    「白煙是液態的小水滴。」
  - 學生提出的證據：  
    「因為水蒸氣凝結成小水滴後，就會形成可見的霧氣。」
  - 預期輸出：  
    是  
  (證據正確描述了凝結過程，且可以支持主張)

  #### 範例二
  - 學生的主張：  
    「白煙是固態的冰晶。」
  - 學生提出的證據：  
    「我覺得因為外面氣溫低到 8 度，就會直接形成結冰。」
  - 預期輸出：  
    否  
  (雖然天氣冷，但 8 度並不足以使水汽直接形成冰晶；證據不夠正確，也無法支撐主張)

  #### 範例三
  - 學生的主張：  
    「白煙是氣態的水蒸氣。」
  - 學生提出的證據：  
    「看起來是白色的煙，應該是蒸氣吧。」
  - 預期輸出：  
    否  
  (「白煙」肉眼可見通常不是純水蒸氣；證據模糊且與正確事實不符，無法有力支持主張)

  #### 範例四
  - 學生的主張：  
    「白煙其實是液態的小水滴。」
  - 學生提出的證據：  
    「用玻璃靠近嘴巴時，會有水珠凝結在表面，表示是液態。」
  - 預期輸出：  
    是  
  (凝結成水珠的事實正確，且能明確支持「白煙是液態水滴」的主張)
  `,
  diagnoseEvidenceSystemPrompt:`
  系統角色：
  你是一個國小自然科學的教學專家 AI。你的任務是閱讀學生在下列對話中提出的主張與證據，並判斷：
  1. 證據是否為正確、符合科學或事實的資訊
  2. 該證據能否合理支持並推理出學生提出的主張
  3. 最後給學生一段 70 字以內的評估回應，只需指出錯誤或不足之處，不要告訴學生答案，也不需進一步教學引導。

  請依照以下步驟進行：
  1. 閱讀並理解學生的主張與證據。
  2. 依據上述兩點進行評估：
    (a) 若證據與事實不符，或無法支持主張，請於回應中指出。
    (b) 若證據正確且合理支撐主張，請於回應中肯定。
  3. 產生一段不超過 70 字的回應即可，不需額外教學或引導。

  **範例輸出格式（請勿包含這行說明文字）**：
  <評估回應，不超過 70 字>

  ### 範例對話情境及預期回應
  #### 範例一
  - 學生主張：
    「白煙是液態的小水滴。」
  - 學生證據：
    「用冷玻璃靠近嘴巴時會凝結成水珠，所以它應該是液態。」
  - 預期回應：
    「不錯！你的觀察能說明白煙的確是水滴凝結，證據能支持你的主張。」

  #### 範例二
  - 學生主張：
    「白煙是固態冰晶。」
  - 學生證據：
    「外面只有攝氏 8 度，就有可能直接結冰。」
  - 預期回應：
    「目前溫度尚不足以形成冰晶，且白煙看似不具固態形狀，證據無法支撐你的主張。」

  #### 範例三
  - 學生主張：
    「白煙是氣態水蒸氣。」
  - 學生證據：
    「看到白色霧狀就一定是水蒸氣。」
  - 預期回應：
    「水蒸氣本身是透明的，此證據不夠正確也無法支持主張。」  
  `,
  reviseEvidenceWithRightClaim:`
  Instruction:
    1. 你是一位國小自然科學的教育專家，你的目標是協助學生修正針對「白煙是液態的小水滴」的不足或錯誤證據，並建構正確的「水的三態」與「三態與溫度的關係」概念。
    2. 每一次只執行一種 Action。
    3. 回答須在 70 字以內。

    學生的主張：[STUDENT_CLAIM]
    學生的證據：[STUDENT_EVIDENCE]

    Action:
        - 喚起相關知識：詢問或確認學生對水的三態及其轉變條件的了解，檢查「溫度」與「狀態改變」等先備概念是否充分。
        - 引導思考現象：透過提問讓學生解釋，在 8 度的情況下，呼出的水氣為何會凝結成可見的霧滴。
        - 協助修正並補充證據：透過「反駁學生的不足或錯誤之處」讓學生理解證據問題，並強調水的液態特徵與溫度影響，協助學生補充或修正。
        - 給予正向回饋並深化：當學生修正或補充證據後，給予肯定並鼓勵他們繼續思考或舉例。

    Example:
    ### 喚起相關知識
    - 學生回答：「我知道水有三態，但不懂為什麼 8 度就能變成水滴」
    - 預期輸出：
    你很棒！那你認為較低溫時，氣態水會怎麼變化呢？

    ### 引導思考現象
    - 學生回答：「因為天氣冷，水蒸氣變成白煙？但不確定為什麼是霧狀」
    - 預期輸出：
    好問題！想想在攝氏 8 度時，哪些原因會讓水分子凝結成看得見的小水滴？

    ### 協助修正並補充證據
    - 學生回答：「我覺得溫度低，可能就直接變成冰。」
    - 預期輸出：
    你的說法有些誤解，8 度還不足以形成冰，請想想水滴凝結的條件是什麼？

    ### 給予正向回饋並深化
    - 學生回答：「原來如此！所以其實是液態，而且我可以用玻璃測試。」
    - 預期輸出：
    很好！用凝結在玻璃表面的現象，就能支持你的結論。
  `,
  reviseEvidenceWithRightClaimEvaluation:`
  系統角色：
  你是一個只輸出「是」或「否」的API。你的任務是判斷在以下情境中，學生是否已經理解先前的證據不足或錯誤，並能建構正確的「水的三態」與「三態與溫度的關係」概念。只需生成「是」或「否」，不需要做任何進一步解釋或回饋。

  寒流來了，氣溫降到攝氏 8 度。大雄、小夫和胖虎在聊天時發現，他們說話時嘴巴前會呼出陣陣白煙。
  他們對這個現象有不同的看法：
      小夫的主張：白煙是液態的小水滴。
      胖虎的主張：白煙是固態的冰晶。
      大雄的主張：白煙是氣態的水蒸氣。

  請依照以下步驟進行：
  1. 閱讀學生的對話與內容。
  2. 參考學生目前的主張和證據
  學生的主張：[STUDENT_CLAIM]
  學生的證據：[STUDENT_EVIDENCE]
  3. 檢查下列條件是否都成立：
    (a) 學生已經理解並承認先前提出的不足或錯誤之處。
    (b) 學生已正確運用「水的三態」與「溫度影響」等概念，並能提出合理解釋。
  3. 若以上兩個條件都成立，判定為「是」；否則判定為「否」。
  4. 請只輸出「是」或「否」作為最後回答。

  **範例輸出格式（請勿包含這行說明文字）**：
  是

  ### 範例對話情境及預期輸出
  #### 範例一
  - 學生先前證據不足，但現在表示：
    「我發現之前的證據沒說明凝結原理。其實在 8 度時，水氣凝結成水滴才是白煙的原因。」
  - 預期輸出：
    是  
  (學生已承認舊證據有誤並運用正確水的三態概念)

  #### 範例二
  - 學生先前證據不足，但現在表示：
    「因為天氣冷，所以我想白煙應該就是小水滴啊？」
  - 預期輸出：
    否  
  (學生並未明確承認舊證據不足，也沒展現對『三態與溫度關係』的深入理解)

  #### 範例三
  - 學生先前證據不足，但現在表示：
    「我知道 8 度不夠低到結冰，所以氣態水只能凝結成小水滴而不是冰晶。」
  - 預期輸出：
    是  
  (學生已釐清之前的不足，且能正確運用水的三態與溫度關係做解釋)
  `,
  reviseEvidenceWithWrongClaim:`
  Instruction:
    1. 你是一位國小自然科學的教育專家，請專注在協助學生意識到「白煙是冰晶」或「白煙是氣態水蒸氣」的主張是錯誤的，以及了解他們在證據中可能存在的迷思、錯誤或不足之處，並逐步引導他們轉變為「白煙是液態的小水滴」的正確觀念。
    2. 每一次只執行一種 Action。
    3. 回答須在 70 字以內。

    學生的主張：[STUDENT_CLAIM]
    學生的證據：[STUDENT_EVIDENCE]

    Action:
        - 質疑固態主張：如果學生認為是冰晶，透過提問引導學生比較冰晶與白煙的特性，引發疑問。
        - 質疑氣態主張：如果學生認為是水蒸氣，透過溫度與可見度的問題，說明其與白煙不同。
        - 質疑證據：如果學生提出的證據有迷思或錯誤之處，透過提問或提醒，讓學生反思證據是否能真正支持主張。
        - 引導認知衝突：在學生猶豫時，運用生活化情境或科學原理，強調錯誤主張或證據中無法解釋的現象。
        - 協助學生自我修正：鼓勵學生根據三態特性修正想法，進一步接近「白煙是液態的小水滴」的結論。
        - 給予正向回饋並深化：當學生開始接受正確觀點，肯定其思考並詢問是否需要更多解釋或舉例。

    Example:
    ### 質疑固態主張
    - 學生回答：「我覺得白煙是冰晶」
    - 預期輸出：
    冰晶有固定形狀，你覺得這些白煙有像冰塊般堅硬嗎？

    ### 質疑氣態主張
    - 學生回答：「應該是水蒸氣吧？」
    - 預期輸出：
    純水蒸氣是看不見的，那你說的白煙怎麼能看得到呢？

    ### 質疑證據
    - 學生回答：「因為外面只有攝氏 8 度，所以一定會結冰啊。」
    - 預期輸出：
    8 度還沒到零下，你覺得這樣的證據能證明白煙是冰晶嗎？

    ### 引導認知衝突
    - 學生回答：「我還是不確定」
    - 預期輸出：
    想想冰需要更低溫才形成，氣態又透明，這些點哪裡矛盾呢？

    ### 協助學生自我修正
    - 學生回答：「好像真的不是冰晶也不是氣體…」
    - 預期輸出：
    太好了！那麼它更像凝結的液態小水滴，你同意嗎？

    ### 給予正向回饋並深化
    - 學生回答：「我懂了，是小水滴」
    - 預期輸出：
    很棒的轉變！看來你的立場已經有所改變。
  `,
  reviseEvidenceWithWrongClaimEvaluation:`
  系統角色：
  你是一個只輸出「是」或「否」的API。你的任務是判斷在
  """
  寒流來了，氣溫降到攝氏 8 度。大雄、小夫和胖虎在聊天時發現，他們說話時嘴巴前會呼出陣陣白煙。他們對這個現象有不同的看法：

  小夫的主張：白煙是液態的小水滴。
  胖虎的主張：白煙是固態的冰晶。
  大雄的主張：白煙是氣態的水蒸氣。
  """
  的情境中，學生的「證據」是否為正確的事實，且能夠支持他提出的主張。只需生成「是」或「否」，不需要做任何進一步解釋或回饋。

  學生的主張：[STUDENT_CLAIM]
  學生的證據：[STUDENT_EVIDENCE]

  請依照以下步驟進行：
  1. 閱讀學生的對話與內容。
  2. 檢查下列條件是否都成立：
    (a) 學生提出的「證據」是否是正確、符合科學或事實的資訊？
    (b) 該證據能否合理支持並推理出學生所提出的主張？
  3. 若以上兩個條件都成立，判定為「是」；否則判定為「否」。
  4. 請只輸出「是」或「否」作為最後回答。

  **範例輸出格式（請勿包含這行說明文字）**：
  是

  ### 範例對話情境及預期輸出
  #### 範例一
  - 學生的主張：  
    「白煙是液態的小水滴。」
  - 學生提出的證據：  
    「因為水蒸氣凝結成小水滴後，就會形成可見的霧氣。」
  - 預期輸出：  
    是  
  (證據正確描述了凝結過程，且可以支持主張)

  #### 範例二
  - 學生的主張：  
    「白煙是固態的冰晶。」
  - 學生提出的證據：  
    「我覺得因為外面氣溫低到 8 度，就會直接形成結冰。」
  - 預期輸出：  
    否  
  (雖然天氣冷，但 8 度並不足以使水汽直接形成冰晶；證據不夠正確，也無法支撐主張)

  #### 範例三
  - 學生的主張：  
    「白煙是氣態的水蒸氣。」
  - 學生提出的證據：  
    「看起來是白色的煙，應該是蒸氣吧。」
  - 預期輸出：  
    否  
  (「白煙」肉眼可見通常不是純水蒸氣；證據模糊且與正確事實不符，無法有力支持主張)

  #### 範例四
  - 學生的主張：  
    「白煙其實是液態的小水滴。」
  - 學生提出的證據：  
    「用玻璃靠近嘴巴時，會有水珠凝結在表面，表示是液態。」
  - 預期輸出：  
    是  
  (凝結成水珠的事實正確，且能明確支持「白煙是液態水滴」的主張)
  `
};

// ------------------- 初始學習狀態 -------------------
function initUserIfNeeded(userId) {
  if (!userState[userId]) {
    // 若此使用者第一次進來，初始化
    userState[userId] = {
      learningState: "claim_stage",
      conversationHistory: [
        {
          role: "assistant",
          content: `
            很好！既然大家已經理解了「主張」、「證據」和「推理」，我們來做一個有趣的情境任務！

            寒流來了，氣溫降到攝氏 8 度。大雄、小夫和胖虎在聊天時發現，
            他們說話時嘴巴前會呼出陣陣白煙。他們對這個現象有不同的看法：
    
            小夫的主張：白煙是液態的小水滴。
            胖虎的主張：白煙是固態的冰晶。
            大雄的主張：白煙是氣態的水蒸氣。
    
            我們來用 CER 論證的方法來分析，看看你們的觀察、證據和推理是什麼。
            請思考一下，誰的主張是正確的？
          `
        }
      ],
      claim: "",
      evidence: "",
      reasoning: "",
      // ...可以再加更多欄位
    };
  }
}


// ------------------- 核心 API -------------------
app.post("/api/chat", async (req, res) => {
  /**
   * 前端可以傳：userId, message, model, currentClaim, currentEvidence 等
   * - userId: 辨識使用者
   * - message: 使用者這次對話
   * - model: 選擇用 chatgpt 或 ollama
   * - currentClaim/currentEvidence: 若使用者在前端填寫了某些主張或證據，可一併帶上
   */
  const { userId, message, model, currentClaim, currentEvidence } = req.body;
  console.log(message);
  // 1. 初始化使用者（若尚未建立）
  initUserIfNeeded(userId);
  
  // 2. 更新自訂欄位（若前端傳了新的值）
  if (typeof currentClaim === "string" && currentClaim.trim() !== "") {
    userState[userId].claim = currentClaim;
  }
  if (typeof currentEvidence === "string" && currentEvidence.trim() !== "") {
    userState[userId].evidence = currentEvidence;
  }

  // 3. 取出此使用者的狀態資料
  const state = userState[userId];
  console.log('claim:', state.claim);
  console.log('evidence:', state.evidence);
  // 4. 將使用者訊息加入對話歷史
  state.conversationHistory.push({ role: "user", content: message });

  //const { message, model } = req.body;

  let finalResponse = "";

  try {
    if (state.learningState === "claim_stage") {
      if (currentClaim) {
        finalResponse = `很好喔，你的主張是「${currentClaim}」，可以試著提出支持此主張的證據嗎？`;
        state.learningState = "evidence_stage_ask_evidence";
      } else {
        finalResponse = "請選擇你的主張，我們才能進一步討論喔！";
      }
    } 
    else if (state.learningState === "evidence_stage_ask_evidence") {
      if (message.includes("我不知道可以提出什麼證據...")) {
        state.learningState = "evidence_stage_ask_evidence_twice";
        finalResponse = "沒關係！證據就是支持主張的資訊，可以是「生活經驗」、「科學原理」或是「課本中的內容」等等...。你要不要試著回想生活經驗，或是從課本學到的知識來找找看有沒有可以支持這個主張的證據？";
      } else if (currentEvidence) {
        // 學生提供了證據，呼叫 evaluateEvidenceSystemPrompt 進行判斷
        state.evidence = currentEvidence;
        // 先將對話歷史 + 本次提示
        const evaluateResponseRaw = await generateResponse(
          [
            {
              role: "user",
              content: `學生的主張：${state.claim}\n學生的證據：${state.evidence}`
            }
          ],
          prompts.evaluateEvidenceSystemPrompt,
          model
        );

        const evaluateResult = evaluateResponseRaw
        .replace(/\s/g, "")   // 把所有空白字元(含換行)都去掉
        .trim();

        if (evaluateResult.startsWith("是")) {
          // 證據有效，進入 reasoning_stage
          state.learningState = "reasoning_stage";
          finalResponse = `你提供的證據是：「${currentEvidence}」。看來它能有效支持「${state.claim}」，讓我們進一步思考推理吧！`;
        } else {
          // 證據無效，呼叫 DIAGNOSE_prompt
          state.learningState = "evidence_stage_revise_evidence"
          const diagnoseResponseRaw = await generateResponse(
            [
              {
                role: "user",
                content: `學生的主張：${state.claim}\n學生的證據：${state.evidence}`
              }
            ],
            prompts.diagnoseEvidenceSystemPrompt,
            model
          );
          finalResponse = diagnoseResponseRaw.trim();
          // 可以考慮要不要改變 learningState
          // state.learningState = "evidence_stage_remedial"; // 例如回到補救教學
        }
      } else {
        finalResponse = "請試著提供一個證據來支持你的主張！";
      }
    }
    else if (state.learningState === "evidence_stage_ask_evidence_twice") {
      if (message.includes("我還是不知道...")) {
        if (state.claim === "白煙是液態的小水滴") {
          state.learningState = "evidence_stage_guild_right_claim";
          const assistantResponse = await generateResponse(
            state.conversationHistory,
            prompts.guildEvidenceWithRightClaimSystemPrompt,
            model
          );
          finalResponse = assistantResponse.trim();
        } else if (state.claim === "白煙是固態的冰晶" || state.claim === "白煙是氣態的水蒸氣") {
          // state.learningState = "evidence_stage_guild_wrong_claim";
          // finalResponse = "好，我們來檢視你的主張，看看是否需要修正。";
          // 新增：呼叫 guildClaimWithWrongClaimSystemPrompt，引導學生改變想法
          state.learningState = "evidence_stage_guild_wrong_claim";
          const assistantResponse = await generateResponse(
            state.conversationHistory,
            prompts.guildClaimWithWrongClaimSystemPrompt,
            model
          );
          finalResponse = assistantResponse.trim();
        } else {
          finalResponse = "請試著再想想你的主張與證據！";
        }
      }
    }
    else if (state.learningState === "evidence_stage_guild_right_claim") {   
      // 將每個對話的 role:content 合併成一個字串，以換行符號分隔
      let mergedContent = state.conversationHistory
      .map(item => `${item.role}:${item.content}`)
      .join('\n');
      // 建立新的陣列，只有一個物件，並將上面合併後的內容放到 content
      const textToBeEval = [
        {
          role: 'user',
          content: mergedContent
        }
      ]
      const rubricResponseRaw = await generateResponse(
        textToBeEval,
        prompts.evaluateUnderstandingSystemPrompt,
        model
      );
      console.log(textToBeEval)
      const rubricResponse = rubricResponseRaw.trim();

      let finalJson = rubricResponse;
      let needGuild = false;

      try {
        // 使用正則表達式提取 {} 內的 JSON 內容
        const match = rubricResponse.match(/\{[\s\S]*\}/);
        
        if (match) {
          const jsonString = match[0]; // 擷取 {} 內的 JSON 內容
          const parsed = JSON.parse(jsonString); // 解析 JSON
      
          // 計算總分
          const totalScore = (parsed.score_concept || 0) + 
                             (parsed.score_mechanism || 0) + 
                             (parsed.score_example || 0);
          // 如果分數 < 2，就需要額外教學
          if (totalScore < 2) {
            needGuild = true;
          }
        } else {
          console.warn("未找到有效的 JSON 內容");
        }
      } catch (e) {
        // 如果 JSON 解析失敗，就原樣返回
        console.error("JSON parse error", e);
      }

      // 如果需要額外教學，就呼叫 guildRightClaimSystemPrompt
      if (needGuild) {
        const guildResponseRaw = await generateResponse(
          state.conversationHistory,
          prompts.guildEvidenceWithRightClaimSystemPrompt,
          model
        );
        const guildResponse = guildResponseRaw.trim();
        finalResponse = guildResponse;
      } else {
        finalResponse = `看來你對「水的三態和變化」有更深的理解了，再次針對${state.claim}提出證據看看吧`; // 如果不用額外教學，就只回傳 JSON

        state.learningState = "evidence_stage_ask_evidence";
      }
    }
    else if (state.learningState === "evidence_stage_guild_wrong_claim") {
      let mergedContent = state.conversationHistory
      .map(item => `${item.role}:${item.content}`)
      .join('\n');
      // 建立新的陣列，只有一個物件，並將上面合併後的內容放到 content
      const textToBeEval = [
        {
          role: 'user',
          content: mergedContent
        }
      ]
      const stanceResponseRaw = await generateResponse(
        textToBeEval,
        prompts.evaluateClaimSystemPrompt,
        model
      );
      const stanceResponse = stanceResponseRaw
        .replace(/\s/g, "")   // 把所有空白字元(含換行)都去掉
        .trim();
      // 如果評估結果為「否」，再進行引導
      if (stanceResponse.startsWith("否")) {
        const assistantResponse = await generateResponse(
          state.conversationHistory,
          prompts.guildClaimWithWrongClaimSystemPrompt,
          model
        );
        finalResponse = assistantResponse.trim();
      } else {
        // 學生已經改變想法
        finalResponse = "太好了！你已經接受了白煙是液態的小水滴的概念。";
        // 可以讓學習階段到下一步或重新設定
        state.learningState = "claim_stage";
      }
    }
    else if (state.learningState === "evidence_stage_revise_evidence") {
      // 根據前端傳入的 message 進行判斷
      if (message.includes("我可以再自己想想看")) {
        // 切換回 evidence_stage_ask_evidence，讓學生重新想證據
        state.learningState = "evidence_stage_ask_evidence";
        finalResponse = "好的，你可以再思考一下，看看還有沒有其他證據能支持你的主張。";
      } else if (message.includes("我需要幫助和引導")) {
        // 根據主張決定下一個階段
        if (state.claim === "白煙是液態的小水滴") {
          state.learningState = "evidence_stage_revise_evidence_right_claim";
          // 將 [STUDENT_CLAIM] 與 [STUDENT_EVIDENCE] 替換為學生的主張與證據
          let revisedPrompt = prompts.reviseEvidenceWithRightClaim
          .replace("[STUDENT_CLAIM]", state.claim)
          .replace("[STUDENT_EVIDENCE]", state.evidence);

        const assistantResponse = await generateResponse(
          state.conversationHistory,
          revisedPrompt,
          model
        );

        finalResponse = assistantResponse.trim();
        // finalResponse = "好的，讓我們一起來看看如何強化『白煙是液態小水滴』的證據。";

        } else if (state.claim === "白煙是氣態的水蒸氣" || state.claim === "白煙是固態的冰晶") {
          state.learningState = "evidence_stage_revise_evidence_wrong_claim";
          let revisedPrompt = prompts.reviseEvidenceWithRightClaim
          .replace("[STUDENT_CLAIM]", state.claim)
          .replace("[STUDENT_EVIDENCE]", state.evidence);

          const assistantResponse = await generateResponse(
            state.conversationHistory,
            revisedPrompt,
            model
          );
          finalResponse = assistantResponse.trim();
          // finalResponse = "好的，讓我們來思考一下是否需要修正你的主張，以及該怎麼找更適切的證據。";
        } else {
          // 其它情況，預設回覆
          finalResponse = "好的，你想從哪方面開始進一步討論呢？";
        }
      } else {
        // 未符合任何指令時
        finalResponse = "請問你想要再想想看，或需要幫助和引導呢？";
      }
    }
    else if (state.learningState === "evidence_stage_revise_evidence_right_claim") {
      // 1. 先用 reviseEvidenceWithRightClaimEvaluation prompt 來判斷學生是否已經修正完畢
    
      // 把 [STUDENT_CLAIM], [STUDENT_EVIDENCE] 替換到 prompt 中
      let evaluationPrompt = prompts.reviseEvidenceWithRightClaimEvaluation
        .replace("[STUDENT_CLAIM]", state.claim)
        .replace("[STUDENT_EVIDENCE]", state.evidence);
    
      const evaluationResponseRaw = await generateResponse(
        state.conversationHistory,
        evaluationPrompt,
        model
      );
    
      // 2. 只輸出「是」或「否」
      const evaluationResult = evaluationResponseRaw.replace(/\s/g, "").trim();
    
      if (evaluationResult.startsWith("是")) {
        // 3. 已完成修正，可前往下一階段，例如 reasoning_stage
        finalResponse = "太好了！看來你對此情境有更多的理解了，可以再提出一次證據嗎？";
        state.learningState = "evidence_stage_ask_evidence"; 
        // 或任何你想要的階段
      } else {
        // 4. 若結果為「否」，則保持在 evidence_stage_revise_evidence_right_claim
        //    並再次使用 reviseEvidenceWithRightClaim prompt 引導學生
        const revisedPrompt = prompts.reviseEvidenceWithRightClaim
          .replace("[STUDENT_CLAIM]", state.claim)
          .replace("[STUDENT_EVIDENCE]", state.evidence);
    
        const assistantResponse = await generateResponse(
          state.conversationHistory,
          revisedPrompt,
          model
        );
        finalResponse = assistantResponse.trim();
        // 保持在同一階段
        state.learningState = "evidence_stage_revise_evidence_right_claim";
      }
    }
    else if (state.learningState === "evidence_stage_revise_evidence_wrong_claim") {
      let evaluationPrompt = prompts.reviseEvidenceWithWrongClaimEvaluation
        .replace("[STUDENT_CLAIM]", state.claim)
        .replace("[STUDENT_EVIDENCE]", state.evidence);
      console.log(state.conversationHistory.slice(-3));
      const evaluationResponseRaw = await generateResponse(
        state.conversationHistory.slice(-3),
        evaluationPrompt,
        model
      );
      const evaluationResult = evaluationResponseRaw.replace(/\s/g, "").trim();
    
      if (evaluationResult.startsWith("是")) {
        state.learningState = "claim_stage";
        finalResponse = "太好了！你已經修正了先前的想法，現在讓我們回到主張階段重新整理吧。";
      } else {
        let revisePrompt = prompts.reviseEvidenceWithWrongClaim
          .replace("[STUDENT_CLAIM]", state.claim)
          .replace("[STUDENT_EVIDENCE]", state.evidence);
    
        const assistantResponse = await generateResponse(
          state.conversationHistory,
          revisePrompt,
          model
        );
        finalResponse = assistantResponse.trim();
      }
    }
    
    // 記錄 AI 回應到對話歷史
    state.conversationHistory.push({ role: "assistant", content: finalResponse });
    console.log(state.conversationHistory);
    // 回傳給前端
    res.json({
      response: finalResponse,
      nextState: state.learningState,
      userClaim: state.claim,
      userEvidence: state.evidence
    });
  
  } catch (error) {
    console.error("Error during AI process:", error);
    res.status(500).json({ error: "Something went wrong." });
  }
});

// 封裝：根據對話和 prompt 呼叫對應的模型
async function generateResponse(conversation, prompt, model) {
    if (model === "chatgpt") {
      return queryOpenAI(conversation, prompt);
    } else {
      return queryOllama(conversation, prompt);
    }
  }
  
// ------------------- 呼叫 OpenAI API -------------------
async function queryOpenAI(conversation, prompt) {
    const response = await fetch(openAiApiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAiApiKey}`
      },
      body: JSON.stringify({
        model: openAiModel,
        messages: [
          { role: "system", content: prompt },
          ...conversation
        ],
        temperature: 0.7
      })
    });
    console.log("query LLM", prompt, conversation);
    const data = await response.json();
    console.log("query reply:", data.choices[0].message.content);
    return data.choices[0].message.content;
  }

// 呼叫 Ollama API
async function queryOllama() {
    const formattedPrompt = conversationHistory.map(msg => 
        `${msg.role === "system" ? "System" : msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`
    ).join("\n") + "\nAssistant:";

    const response = await fetch(ollamaApiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: ollamaModel, prompt: formattedPrompt, stream: false })
    });

    const data = await response.json();
    return data.response;
}

// 啟動伺服器
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});