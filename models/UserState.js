import mongoose from "mongoose";

// 定義資料結構
const userSchema = new mongoose.Schema({
  userId: { type: String, required: true, unique: true },
  learningState: { type: String, default: "claim_stage" },
  claim: { type: String, default: "" },
  evidence: { type: String, default: "" },
  reasoning: { type: String, default: "" },
  conversationHistory: [
    {
      role: { type: String },
      content: { type: String }
    }
  ]
});

// 產生 Model (UserState)
const UserState = mongoose.model("UserState", userSchema);

export default UserState;
