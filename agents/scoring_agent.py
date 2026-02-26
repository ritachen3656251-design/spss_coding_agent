"""
智能打分 Agent：记忆、决策、行动、反思
"""
import os
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.case_repository import CaseRepository
from agents.memory import AgentMemory
from tools.keyword_matcher import KeywordMatcher
from tools.llm_client import LLMClient
import config


class ScoringAgent(BaseAgent):
    """
    智能打分 Agent：
    - 记忆：记住之前的决策经验
    - 决策：根据观察选择策略（ReAct 或规则）
    - 行动：执行打分并反思结果
    """

    def __init__(self, rubric: str, scenario: str = None, use_iteration: bool = False):
        super().__init__(name="ScoringAgent", llm_client=LLMClient(model_type="qwen"))
        self.rubric = rubric
        self.scenario = scenario

        self.memory = AgentMemory()
        self.case_repo = CaseRepository()

        has_deepseek_key = config.DEEPSEEK_API_KEY or os.getenv("OPENAI_API_KEY")
        self.secondary_client = (
            LLMClient(model_type="deepseek")
            if (config.ENABLE_DUAL_MODEL and has_deepseek_key)
            else None
        )
        self.keyword_matcher = (
            KeywordMatcher() if config.ENABLE_KEYWORD_FALLBACK else None
        )

    def process(self, text: str, respondent_id: str = None) -> dict:
        """
        完整的Agent流程：ReAct 循环或规划流程，集成案例库学习
        """
        self.log(f"开始处理：'{text[:30]}...'")

        if config.ENABLE_REACT:
            result = self._react_loop(text)
        else:
            observation = self._observe(text)
            plan = self._simple_plan(text, observation)
            result = self._execute_plan(text, plan, observation)

        reflection = self._reflect(text, result, {"plan": [], "reasoning": ""})
        result["reflection"] = reflection
        self.log(f"反思：{reflection['insight']}")

        # 存储有效经验，形成闭环
        if result.get("confidence", 0) >= 0.7 and result.get("score") is not None:
            self.memory.add_case({
                "text": text,
                "strategy": result["strategy"],
                "score": result["score"],
                "confidence": result["confidence"],
            })

        self.memory.record_decision(
            text,
            result["strategy"],
            result.get("score"),
            result.get("confidence", 0.0),
        )

        # 检查是否为疑难案例并记录到案例库
        if self._should_record_as_difficult(result):
            case_data = {
                "id": respondent_id or "unknown",
                "text": text,
                "score": result.get("score"),
                "confidence": result.get("confidence", 0),
                "timestamp": datetime.now().isoformat(),
                "iteration_count": result.get("iteration_count", 1),
            }

            details = result.get("details", {})
            if "secondary" in details:
                primary_score = details["primary"].get("score")
                secondary_score = details["secondary"].get("score")

                if primary_score != secondary_score:
                    case_data["conflict_details"] = {
                        "primary_score": primary_score,
                        "secondary_score": secondary_score,
                        "primary_confidence": details["primary"].get("confidence"),
                        "secondary_confidence": details["secondary"].get("confidence"),
                    }

            added = self.case_repo.add_difficult_case(case_data)
            if added:
                self.log(f"✓ 已记录为疑难案例（编号:{respondent_id or 'unknown'}）")

        return result

    def _simple_plan(self, text: str, observation: dict) -> dict:
        """规则规划：LLM 规划简化内联"""
        plan = []
        reasoning = []

        if len(text) < 10 and self.scenario:
            plan.append("use_context")
            reasoning.append("文本过短，需要情景信息")

        ambiguous_words = ["想", "可能", "好像", "应该"]
        if any(word in text for word in ambiguous_words):
            plan.append("prioritize_dual_model")
            reasoning.append("包含模糊表述，可能需要多模型验证")

        strategy_advice = self.memory.get_strategy_advice()
        if strategy_advice == "dual_model_recommended":
            plan.append("prioritize_dual_model")
            reasoning.append("历史数据显示双模型策略效果好")

        if observation.get("is_boundary"):
            plan.append("record_to_memory")
            reasoning.append("边界 case，需记录以改进未来决策")

        if not plan:
            plan = ["standard_scoring"]
            reasoning = ["常规打分流程"]

        return {"plan": plan, "reasoning": " → ".join(reasoning)}

    def _check_context_needed(self, answer_text: str) -> dict:
        """判断是否需要补充情景信息（原 ContextAgent 逻辑）"""
        if not self.scenario:
            return {"needs_context": False, "reason": None, "enhanced_prompt": None}

        low_info_signals = [
            len(answer_text) < 10,
            answer_text.count("，") == 0 and answer_text.count("。") == 0,
            not any(
                kw in answer_text for kw in ["以为", "想", "没发现", "拿", "看"]
            ),
        ]

        if any(low_info_signals):
            enhanced_prompt = f"""
# 原始情景（帮助你理解背景）
{self.scenario}

# 被试的回答
"{answer_text}"

# 评分提示
这个回答虽然简短，但请结合原始情景判断：
1. 如果回答暗示了"误以为猫是围巾"这个错误信念 → 2分
2. 如果只是提到了想法或意图，但没有明确错误信念 → 1分
3. 如果只是描述物理动作 → 0分

即使回答很短，也要尽量从情景中推断其含义，而不是轻易标记为"信息不足"。
"""
            return {
                "needs_context": True,
                "reason": "回答信息量较少，需要情景辅助",
                "enhanced_prompt": enhanced_prompt,
            }

        return {"needs_context": False, "reason": None, "enhanced_prompt": None}

    def _observe(self, text: str) -> dict:
        """观察阶段：分析文本特征"""
        keywords = ["以为", "没发现", "想", "拿"]
        ambiguous_words = ["好像", "可能", "应该"]
        has_kw = any(kw in text for kw in keywords)
        is_short = len(text) < 10
        is_ambiguous = any(w in text for w in ambiguous_words)
        return {
            "length": len(text),
            "has_keywords": has_kw,
            "is_short": is_short,
            "is_ambiguous": is_ambiguous,
            "is_boundary": is_short or is_ambiguous,
            "summary": f"长度{len(text)}字，{'包含' if has_kw else '不含'}关键词",
        }

    def _react_loop(self, text: str) -> dict:
        """ReAct 循环：思考→行动→观察→检查"""
        max_iter = config.REACT_MAX_ITERATIONS
        target_conf = config.REACT_CONFIDENCE_TARGET

        observation = self._observe(text)
        similar_cases = self.memory.retrieve_similar(text)
        best_from_history = self.memory.get_best_strategy_from_cases(similar_cases)
        if similar_cases and best_from_history:
            self.log(f"检索到 {len(similar_cases)} 条相似经验，建议策略：{best_from_history}")

        state = {
            "text": text,
            "observation": observation,
            "tried_actions": [],
            "last_result": None,
            "similar_cases": similar_cases,
            "best_from_history": best_from_history,
        }

        for i in range(max_iter):
            self.log(f"ReAct 第 {i + 1}/{max_iter} 轮")

            thought = self._react_think(state)
            self.log(f"思考：{thought[:80]}...")

            action = self._react_decide_action(thought, state)
            if not action:
                action = "standard_scoring"
            self.log(f"行动：{action}")

            result = self._execute_react_action(action, text, state)
            state["last_result"] = result
            state["tried_actions"].append((action, result))

            if result.get("confidence", 0) >= target_conf:
                self.log(f"置信度达 {result['confidence']:.2f}，结束循环")
                result["iteration_count"] = len(state["tried_actions"])
                return result

        best = max(
            state["tried_actions"],
            key=lambda x: x[1].get("confidence", 0) if x[1] else 0,
        )
        best[1]["iteration_count"] = len(state["tried_actions"])
        return best[1]

    def _react_think(self, state: dict) -> str:
        """Thought：分析当前状态"""
        obs = state["observation"]
        tried = state["tried_actions"]
        last = state.get("last_result")

        history = ""
        if tried:
            history = "已尝试：" + "；".join(
                f"{a}→置信度{r.get('confidence', 0):.2f}"
                for a, r in tried
            )

        system = "你是问卷编码决策专家。请简短分析当前状态并思考下一步策略。"
        user = f"""文本观察：{obs['summary']}
{history if history else "首次尝试"}
{f"上次结果：分数={last.get('score')}，置信度={last.get('confidence', 0):.2f}" if last else ""}

请用1-2句话分析，应选择哪种策略。"""
        out = self.llm_client.call(system, user, max_tokens=150)
        return out.get("response", "") if out.get("success") else ""

    def _react_decide_action(self, thought: str, state: dict) -> str | None:
        """Act：决定下一步行动"""
        tried_actions = [a for a, _ in state["tried_actions"]]
        available = [
            "standard_scoring",
            "context_enhanced" if self.scenario else None,
            "dual_model" if self.secondary_client else None,
        ]
        available = [a for a in available if a and a not in tried_actions]

        if not available:
            return "standard_scoring"

        best_from_history = state.get("best_from_history")
        history_to_action = {
            "use_context": "context_enhanced",
            "prioritize_dual_model": "dual_model",
            "standard_scoring": "standard_scoring",
        }
        history_action = history_to_action.get(best_from_history) if best_from_history else None
        if history_action and history_action in available:
            self.log(f"历史经验提示使用：{history_action}")
            return history_action

        system = """你输出一个 JSON，只包含 action 字段。
可选 action：standard_scoring（标准打分）、context_enhanced（情景辅助）、dual_model（双模型验证）
选择最合适的，且不要重复已尝试的。"""
        user = f"思考：{thought}\n已尝试：{tried_actions}\n可选：{available}\n输出：{{\"action\": \"xxx\"}}"
        out = self.call_llm_with_json(system, user)
        if out.get("success") and out.get("data"):
            action = out["data"].get("action", "")
            if action in available:
                return action
            if action == "context_enhanced" and self.scenario:
                return "context_enhanced"
            if action == "dual_model" and self.secondary_client:
                return "dual_model"
        return available[0]

    def _execute_react_action(self, action: str, text: str, state: dict) -> dict:
        """执行 ReAct 行动"""
        if action == "context_enhanced" and self.scenario:
            return self._score_with_context(text)
        if action == "dual_model" and self.secondary_client:
            return self._score_with_dual_model_first(text)
        primary = self._score_with_primary_model(text)
        if primary["confidence"] < config.CONFIDENCE_THRESHOLD:
            return self._handle_boundary_case(text, primary)
        return {
            "score": primary["score"],
            "confidence": primary["confidence"],
            "reasoning": primary.get("reasoning", ""),
            "strategy": "primary",
            "details": {"primary": primary},
        }

    def _execute_plan(self, text: str, plan: dict, observation: dict) -> dict:
        """执行计划：根据规划选择策略，优先采纳相似历史经验"""
        similar_cases = self.memory.retrieve_similar(text)
        best_from_history = self.memory.get_best_strategy_from_cases(similar_cases)

        if similar_cases and best_from_history:
            self.log(f"检索到 {len(similar_cases)} 条相似经验，建议策略：{best_from_history}")

        # 历史经验优先：若相似案例指向明确策略且具备能力，则采用
        effective_plan = list(plan["plan"])
        if best_from_history and best_from_history not in effective_plan:
            effective_plan.insert(0, best_from_history)

        if "use_context" in effective_plan and self.scenario:
            self.log("执行：使用情景辅助")
            return self._score_with_context(text)

        elif "prioritize_dual_model" in effective_plan and self.secondary_client:
            self.log("执行：优先使用双模型验证")
            return self._score_with_dual_model_first(text)

        else:
            self.log("执行：标准打分流程")
            primary_result = self._score_with_primary_model(text)

            if primary_result["confidence"] < config.CONFIDENCE_THRESHOLD:
                self.log("决策：置信度低，启动多策略验证")
                return self._handle_boundary_case(text, primary_result)
            else:
                return {
                    "score": primary_result["score"],
                    "confidence": primary_result["confidence"],
                    "reasoning": "",
                    "strategy": "primary",
                    "details": {"primary": primary_result},
                }

    def _reflect(self, text: str, result: dict, plan: dict) -> dict:
        """反思阶段：评估决策质量"""
        confidence = result.get("confidence", 0.0)
        if confidence < 0.5:
            insight = "策略未能提升置信度，可能需要人工介入"
        elif confidence >= 0.8:
            insight = "策略有效，置信度高"
        else:
            insight = "策略部分有效，置信度中等"

        should_remember = (
            confidence < 0.6
            or result.get("strategy") == "dual_model_consensus"
        )

        if should_remember:
            edge_cases = self.memory.long_term.setdefault("edge_cases", [])
            edge_cases.append({
                "text": text[:50],
                "score": result.get("score"),
                "strategy": result.get("strategy"),
                "confidence": confidence,
            })
            self.memory.long_term["edge_cases"] = edge_cases[-100:]

        return {"insight": insight, "should_remember": should_remember}

    def _should_record_as_difficult(self, result: dict) -> bool:
        """判断是否应该记录为疑难案例"""
        if result.get("confidence", 1) < 0.7:
            return True

        if "details" in result and "secondary" in result["details"]:
            primary_score = result["details"]["primary"].get("score")
            secondary_score = result["details"]["secondary"].get("score")
            if primary_score != secondary_score:
                return True

        if result.get("iteration_count", 1) > 1:
            return True

        if result.get("score") == 1:
            return True

        return False

    def save_knowledge(self):
        """保存学到的知识"""
        self.case_repo.save_cases()
        self.log("✓ 知识库已保存")

    def _score_with_context(self, text: str) -> dict:
        """使用情景辅助的打分"""
        if not self.scenario:
            return self._score_with_primary_model(text)

        context_check = self._check_context_needed(text)

        if context_check["needs_context"]:
            enhanced_result = self._score_with_enhanced_prompt(
                text, context_check["enhanced_prompt"]
            )
            enhanced_result["strategy"] = "context_enhanced"
            enhanced_result["details"] = {"primary": enhanced_result.copy()}
            return enhanced_result
        else:
            primary_result = self._score_with_primary_model(text)
            return {
                "score": primary_result["score"],
                "confidence": primary_result["confidence"],
                "reasoning": primary_result.get("reasoning", ""),
                "strategy": "primary",
                "details": {"primary": primary_result},
            }

    def _score_with_enhanced_prompt(self, text: str, enhanced_prompt: str) -> dict:
        """使用增强提示词打分"""
        user_message = enhanced_prompt + f"\n\n{self.rubric}"
        system_prompt = """你是严格的心理学问卷编码专家。

你必须输出JSON格式：
{
    "score": 0/1/2,
    "confidence": 0.0-1.0,
    "reasoning": "判分理由",
    "key_evidence": ["支持该分数的关键证据"]
}

只返回JSON，不要其他文字。"""
        result = self.call_llm_with_json(system_prompt, user_message)
        if not result["success"]:
            return {
                "score": None,
                "confidence": 0.0,
                "reasoning": "LLM调用失败",
                "key_evidence": [],
            }
        return result["data"]

    def _score_with_dual_model_first(self, text: str) -> dict:
        """优先使用双模型的策略"""
        primary_result = self._score_with_primary_model(text)
        secondary_result = self._score_with_secondary_model(text)

        if primary_result["score"] == secondary_result["score"]:
            return {
                "score": primary_result["score"],
                "confidence": min(0.95, primary_result["confidence"] + 0.15),
                "reasoning": f"双模型一致：{primary_result.get('reasoning', '')}",
                "strategy": "dual_model_consensus",
                "details": {"primary": primary_result, "secondary": secondary_result},
            }
        else:
            if primary_result["confidence"] > secondary_result["confidence"]:
                return {
                    "score": primary_result["score"],
                    "confidence": primary_result["confidence"],
                    "reasoning": primary_result.get("reasoning", ""),
                    "strategy": "primary_higher_confidence",
                    "details": {"primary": primary_result, "secondary": secondary_result},
                }
            else:
                return {
                    "score": secondary_result["score"],
                    "confidence": secondary_result["confidence"],
                    "reasoning": secondary_result.get("reasoning", ""),
                    "strategy": "secondary_higher_confidence",
                    "details": {"primary": primary_result, "secondary": secondary_result},
                }

    def _score_with_primary_model(self, text: str) -> dict:
        """使用主模型打分，必要时提供情景"""
        if self.scenario:
            context_check = self._check_context_needed(text)
            if context_check["needs_context"]:
                self.log("检测到信息不足，提供情景辅助")
                user_message = context_check["enhanced_prompt"] + f"\n\n{self.rubric}"
            else:
                user_message = f"""# 评分标准
{self.rubric}

# 待评分回答
"{text}"

请评分并说明你的置信度。"""
        else:
            user_message = f"""# 评分标准
{self.rubric}

# 待评分回答
"{text}"

请评分并说明你的置信度。"""

        system_prompt = """你是严格的心理学问卷编码专家。

你必须输出JSON格式：
{
    "score": 0/1/2,
    "confidence": 0.0-1.0,
    "reasoning": "判分理由",
    "key_evidence": ["支持该分数的关键证据"]
}

confidence说明：
- 0.9-1.0: 非常确定，有明确关键词
- 0.7-0.89: 比较确定，逻辑清晰
- 0.5-0.69: 有一定依据，但存在模糊性
- 0.3-0.49: 倾向某个分数，但不太确定
- 0.0-0.29: 完全无法判断

重要：即使信息较少，也要基于情景和上下文给出最合理的判断，不要轻易说"信息不足"。

只返回JSON，不要其他文字。"""

        result = self.call_llm_with_json(system_prompt, user_message)

        if not result["success"]:
            return {
                "score": None,
                "confidence": 0.0,
                "reasoning": "LLM调用失败",
                "key_evidence": [],
            }

        return result["data"]

    def _handle_boundary_case(self, text: str, primary_result: dict) -> dict:
        """处理边界 case：多策略验证"""
        all_results = {"primary": primary_result}

        if config.ENABLE_DUAL_MODEL and self.secondary_client:
            self.log("调用第二模型（DeepSeek）验证...")
            secondary_result = self._score_with_secondary_model(text)
            all_results["secondary"] = secondary_result

            if secondary_result["score"] == primary_result["score"]:
                self.log("双模型结果一致，提升置信度")
                return {
                    "score": primary_result["score"],
                    "confidence": min(primary_result["confidence"] + 0.2, 0.95),
                    "reasoning": f"双模型一致判定：{primary_result.get('reasoning', '')}",
                    "strategy": "dual_model_consensus",
                    "details": all_results,
                }

            self.log(
                f"双模型不一致：Qwen={primary_result['score']}, DeepSeek={secondary_result['score']}"
            )

        if config.ENABLE_KEYWORD_FALLBACK and self.keyword_matcher:
            self.log("使用关键词匹配作为回退策略...")
            keyword_result = self.keyword_matcher.match(text)
            all_results["keyword"] = keyword_result

            if keyword_result["score"] == primary_result["score"]:
                self.log("关键词匹配与主模型一致")
                return {
                    "score": primary_result["score"],
                    "confidence": max(
                        primary_result["confidence"], keyword_result["confidence"]
                    ),
                    "reasoning": f"{primary_result.get('reasoning', '')} (关键词验证: {keyword_result['matched_patterns']})",
                    "strategy": "keyword_consensus",
                    "details": all_results,
                }

        self.log("多策略验证后仍不确定，标记为需人工复核")
        return {
            "score": primary_result["score"],
            "confidence": primary_result["confidence"],
            "reasoning": primary_result.get("reasoning", "") + " [需人工复核]",
            "strategy": "uncertain",
            "details": all_results,
            "needs_human_review": True,
        }

    def _score_with_secondary_model(self, text: str) -> dict:
        """使用第二模型（DeepSeek）打分"""
        original_client = self.llm_client
        self.llm_client = self.secondary_client
        result = self._score_with_primary_model(text)
        self.llm_client = original_client
        return result


def load_rubric(file_path: str = "prompts/scoring_rubric.txt") -> str:
    """读取评分标准文件"""
    encodings = ("utf-8", "gbk")
    last_error = None
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"无法使用 {encodings} 解码文件 {file_path}",
    )
