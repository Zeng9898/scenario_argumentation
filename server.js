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
  guildRightClaimSystemPrompt:
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
    - 學生回答：「我有點懂了，但還想多了解一些」
    - 預期輸出：
    很棒的提問！可以再想想水滴在空氣中的形態，或者要不要用其他實驗來證明？
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
            prompts.guildRightClaimSystemPrompt,
            model
          );
          finalResponse = assistantResponse.trim();
        } else if (state.claim === "白煙是固態的冰晶" || state.claim === "白煙是氣態的水蒸氣") {
          state.learningState = "evidence_stage_guild_wrong_claim";
          finalResponse = "好，我們來檢視你的主張，看看是否需要修正。";
        } else {
          finalResponse = "請試著再想想你的主張與證據！";
        }
      }
    }
    else if (state.learningState === "evidence_stage_guild_right_claim") {
      // 在這裡可以再進一步呼叫 generateResponse，引導學生繼續思考。
      const assistantResponse = await generateResponse(
        state.conversationHistory,
        prompts.guildRightClaimSystemPrompt,
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