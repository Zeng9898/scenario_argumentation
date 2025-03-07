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
  guildClaimWithWrongClaimSystemPrompt:
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
  if (typeof currentClaim === "string" && currentClaim.trim() !== "") {
    userState[userId].evidence = currentEvidence;
  }

  // 3. 取出此使用者的狀態資料
  const state = userState[userId];
  console.log('claim:', state.claim);
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
        finalResponse = "沒關係！你要不要試著回想生活經驗，或是從課本學到的知識來找找看有沒有可以支持這個主張的證據？";
      } else if (currentEvidence) {
        state.evidence = currentEvidence;
        finalResponse = `你提供的證據是：「${currentEvidence}」，我們來看看這個證據是否足夠支持你的主張。`;
        state.learningState = "evaluate_evidence";
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
      // 進行同樣的引導
      const assistantResponse = await generateResponse(
        state.conversationHistory,
        prompts.guildClaimWithWrongClaimSystemPrompt,
        model
      );
      finalResponse = assistantResponse.trim();
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
    console.log("query LLM", prompt.slice(0, 100), conversation);
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