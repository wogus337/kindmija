
import streamlit as st
import requests
import json
import re

# Perplexity AI API 키 (실제 키로 교체 필요)
PERPLEXITY_API_KEY = "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"

# 크레딧 사용량 추적 변수
credit_used = 0
CREDIT_LIMIT = 5

# API 호출에 따른 크레딧 소모량 (예시)
CREDIT_PER_CALL = 0.02  # 예를 들어, 한 호출당 0.02 크레딧 소모


def get_perplexity_response(company_name):
    global credit_used
    if credit_used >= CREDIT_LIMIT:
        return "크레딧 한도 초과로 API 호출이 불가능합니다."

    url = "https://api.perplexity.ai/chat/completions"
    query = f"{company_name}의 최근 실적을 정리해 주고 특이사항이 있다면 알려주세요. 새롭게 발행하는 {company_name} 회사채에 대한 투자를 고려하고 있는데 신용등급 전망과 재무 상황, 업황, 실적 전망을 요약해 주세요. 각종 자료에 대해서는 출처도 함께 명시해 주었으면 좋겠어요"
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

    # 응답 상태 코드 확인
    if response.status_code != 200:
        return f"API 호출 실패: {response.status_code}, 응답: {response.text}", []

    # 응답 JSON 확인
    response_json = response.json()

    if "choices" in response_json and response_json["choices"]:
        content = response_json["choices"][0]["message"]["content"]
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        citations = response_json.get("citations", [])
        return content.strip(), citations
    else:
        return f"API 응답에 'choices' 키가 없습니다. 응답 내용: {response_json}", []


def main():
    st.title("투자정보 조회")

    company_name = st.text_input("기업명을 입력하세요")

    if st.button("정보 조회"):
        if company_name:
            if credit_used < CREDIT_LIMIT:
                with st.spinner("정보를 조회 중입니다..."):
                    result = get_perplexity_response(company_name)
                    if isinstance(result, tuple) and len(result) == 2:
                        content, citations = result
                        st.markdown(content)
                        if citations:
                            st.subheader("출처")
                            for i, citation in enumerate(citations, 1):
                                st.write(f"{i}. {citation}")
                    else:
                        st.write(result)
                st.info(f"현재 사용한 크레딧: ${credit_used:.2f} / ${CREDIT_LIMIT:.2f}")
            else:
                st.warning("크레딧 한도에 도달했습니다. 더 이상 API 호출을 할 수 없습니다.")
        else:
            st.warning("기업명을 입력해주세요.")


if __name__ == "__main__":
    main()