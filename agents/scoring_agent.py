"""
智能打分 Agent：观察 → 规划 → 选择策略 → 执行 → 反思
规则驱动的智能体流程，结合记忆与案例库学习
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
    智能打分 Agent（规则规划模式）：
    1. 观察：分析文本特征
    2. 规划：基于规则生成候选策略及理由
    3. 选择策略：结合历史经验选出最优策略
    4. 执行：调用对应打分流程
    5. 反思：评估决策质量，更新记忆
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
            KeywordMatcher()         if config.ENABLE_KEYWORD_FALLBACK else None
        )

    def process(self, text: str, respondent_id: str = None) -> dict:
        """
        Agent 完整流程：观察 → 规划 → 选择策略 → 执行 → 反思
        """
        self.log(f"开始处理：'{text[:30]}...'")

        # 1. 观察：分析文本特征
        observation = self._observe(text)
        self.log(f"观察：{observation['summary']}")

        # 2. 规划：基于规则生成候选策略
        plan = self._plan(text, observation)
        self.log(f"规划：{plan['reasoning']}")

        # 3. 选择策略：结合历史经验确定最终策略
        strategy = self._select_strategy(text, plan, observation)
        self.log(f"选择策略：{strategy}")

        # 4. 执行：调用对应打分流程
        result = self._execute(text, strategy, observation)

        # 5. 反思：评估决策质量
        reflection = self._reflect(text, result, plan)
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
                "iteration_count": 1,
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

    def _plan(self, text: str, observation: dict) -> dict:
        """
        规划阶段：基于规则生成候选策略列表及理由
        返回：{"candidates": [strategy1, strategy2, ...], "reasoning": str}
        """
        candidates = []
        reasoning = []

        # 规则1：文本过短且有情景 → 优先情景辅助
        if len(text) < 10 and self.scenario:
            candidates.append("use_context")
            reasoning.append("文本过短，需要情景信息")

        # 规则2：包含模糊表述 → 考虑双模型验证
        ambiguous_words = ["想", "可能", "好像", "应该"]
        if any(word in text for word in ambiguous_words):
            if "prioritize_dual_model" not in candidates and self.secondary_client:
                candidates.append("prioritize_dual_model")
            reasoning.append("包含模糊表述，可能需要多模型验证")

        # 规则3：历史经验建议
        strategy_advice = self.memory.get_strategy_advice()
        if strategy_advice == "dual_model_recommended" and self.secondary_client:
            if "prioritize_dual_model" not in candidates:
                candidates.append("prioritize_dual_model")
            reasoning.append("历史数据显示双模型策略效果好")

        # 规则4：边界 case 标记（供反思阶段使用）
        if observation.get("is_boundary"):
            reasoning.append("边界 case，需记录以改进未来决策")

        # 兜底：至少一个策略
        if not candidates:
            candidates = ["standard_scoring"]
            reasoning = ["常规打分流程"]

        return {
            "candidates": candidates,
            "reasoning": " → ".join(reasoning),
            "is_boundary": observation.get("is_boundary", False),
        }

    def _select_strategy(self, text: str, plan: dict, observation: dict) -> str:
        """
        选择策略：结合规划候选与历史经验，选出最终执行的策略
        优先级：历史相似案例建议 > 规划首位
        """
        candidates = plan["candidates"]

        # 检索相似历史经验
        similar_cases = self.memory.retrieve_similar(text)
        best_from_history = self.memory.get_best_strategy_from_cases(similar_cases)

        if similar_cases and best_from_history:
            self.log(f"检索到 {len(similar_cases)} 条相似经验，建议策略：{best_from_history}")

        # 历史建议若在候选列表中，则优先采用
        history_to_strategy = {
            "use_context": "use_context",
            "prioritize_dual_model": "prioritize_dual_model",
            "standard_scoring": "standard_scoring",
        }
        history_strategy = history_to_strategy.get(best_from_history)
        if history_strategy and history_strategy in candidates:
            return history_strategy

        # 否则采用规划首位
        return candidates[0]

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

    def _execute(self, text: str, strategy: str, observation: dict) -> dict:
        """
        执行阶段：根据所选策略调用对应打分流程
        strategy: use_context | prioritize_dual_model | standard_scoring
        """
        if strategy == "use_context" and self.scenario:
            self.log("执行：情景辅助打分")
            return self._score_with_context(text, observation)

        if strategy == "prioritize_dual_model" and self.secondary_client:
            self.log("执行：双模型验证")
            return self._score_with_dual_model_first(text, observation)

        # standard_scoring 或策略不可用时
        self.log("执行：标准打分")
        primary_result = self._score_with_primary_model(text, observation)

        if primary_result["confidence"] < config.CONFIDENCE_THRESHOLD:
            self.log("置信度低，启动边界 case 多策略验证")
            return self._handle_boundary_case(text, primary_result)

        return {
            "score": primary_result["score"],
            "confidence": primary_result["confidence"],
            "reasoning": primary_result.get("reasoning", ""),
            "strategy": "primary",
            "details": {"primary": primary_result},
        }

    def _reflect(self, text: str, result: dict, plan: dict = None) -> dict:
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
        """判断是否应该记录为疑难案例（收紧条件，避免案例库膨胀）"""
        confidence = result.get("confidence", 1)
        score = result.get("score")

        if confidence < config.CONFIDENCE_THRESHOLD:
            return True

        if "details" in result and "secondary" in result["details"]:
            primary_score = result["details"]["primary"].get("score")
            secondary_score = result["details"]["secondary"].get("score")
            if primary_score != secondary_score:
                return True

        if (
            config.RECORD_SCORE1_AS_DIFFICULT
            and score == 1
            and confidence < config.RECORD_SCORE1_CONFIDENCE_MAX
        ):
            return True

        return False

    def save_knowledge(self):
        """保存学到的知识"""
        self.case_repo.save_cases()
        self.log("✓ 知识库已保存")

    def _score_with_context(self, text: str, observation: dict = None) -> dict:
        """使用情景辅助的打分"""
        if not self.scenario:
            return self._score_with_primary_model(text, observation)

        context_check = self._check_context_needed(text)

        if context_check["needs_context"]:
            rubric = self.rubric
            enhanced_result = self._score_with_enhanced_prompt(
                text, context_check["enhanced_prompt"], rubric
            )
            enhanced_result["strategy"] = "context_enhanced"
            enhanced_result["details"] = {"primary": enhanced_result.copy()}
            return enhanced_result
        else:
            primary_result = self._score_with_primary_model(text, observation)
            return {
                "score": primary_result["score"],
                "confidence": primary_result["confidence"],
                "reasoning": primary_result.get("reasoning", ""),
                "strategy": "primary",
                "details": {"primary": primary_result},
            }

    def _score_with_enhanced_prompt(
        self, text: str, enhanced_prompt: str, rubric: str = None
    ) -> dict:
        """使用增强提示词打分（情景辅助）"""
        rubric = rubric or self.rubric
        user_message = enhanced_prompt + f"\n\n# 评分标准\n{rubric}"
        system_prompt = self._build_system_prompt()
        result = self.call_llm_with_json(system_prompt, user_message)
        if not result["success"]:
            return {
                "score": None,
                "confidence": 0.0,
                "reasoning": "LLM调用失败",
                "key_evidence": [],
            }
        return result["data"]

    def _score_with_dual_model_first(self, text: str, observation: dict = None) -> dict:
        """优先使用双模型的策略"""
        primary_result = self._score_with_primary_model(text, observation)
        secondary_result = self._score_with_secondary_model(text, observation)

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

    def _build_system_prompt(self) -> str:
        """构建系统提示词（保持与高 Kappa 时期一致）"""
        return """你是严格的心理学问卷编码专家。

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

    def _score_with_primary_model(self, text: str, observation: dict = None) -> dict:
        """使用主模型打分，必要时提供情景"""
        rubric = self.rubric
        if self.scenario:
            context_check = self._check_context_needed(text)
            if context_check["needs_context"]:
                self.log("检测到信息不足，提供情景辅助")
                user_message = context_check["enhanced_prompt"] + f"\n\n# 评分标准\n{rubric}"
            else:
                user_message = f"""# 评分标准
{rubric}

# 待评分回答
"{text}"

请评分并说明你的置信度。"""
        else:
            user_message = f"""# 评分标准
{rubric}

# 待评分回答
"{text}"

请评分并说明你的置信度。"""

        system_prompt = self._build_system_prompt()
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

    def _score_with_secondary_model(self, text: str, observation: dict = None) -> dict:
        """使用第二模型（DeepSeek）打分"""
        original_client = self.llm_client
        self.llm_client = self.secondary_client
        result = self._score_with_primary_model(text, observation)
        self.llm_client = original_client
        return result


def load_scenario(scenario_dir: str = "prompts") -> str | None:
    """加载情景描述（scenario.txt）"""
    txt_path = os.path.join(scenario_dir, "scenario.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


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
