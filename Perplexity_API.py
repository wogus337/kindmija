
import streamlit as st
import requests
import json

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
    query = f"새롭게 발행하는 {company_name} 회사채에 대한 투자를 고려하고 있는데 신용등급 전망과 재무 상황, 실적 전망을 요약해 주세요. 각종 자료에 대해서는 출처도 함께 명시해 주었으면 좋겠어요"
    payload = {
        "model": "mixtral-8x7b-instruct",
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
        return f"API 호출 실패: {response.status_code}, 응답: {response.text}"

    # 응답 JSON 확인
    response_json = response.json()
    st.write("API 응답:", response_json)  # 디버깅용 출력

    # choices 키가 있는지 확인 후 반환
    if "choices" in response_json and response_json["choices"]:
        credit_used += CREDIT_PER_CALL  # 크레딧 사용량 증가
        return response_json["choices"][0]["message"]["content"]
    else:
        return f"API 응답에 'choices' 키가 없습니다. 응답 내용: {response_json}"


def main():
    st.title("기업 회사채 투자 정보 조회")

    company_name = st.text_input("기업명을 입력하세요")

    if st.button("정보 조회"):
        if company_name:
            if credit_used < CREDIT_LIMIT:
                with st.spinner("정보를 조회 중입니다..."):
                    response = get_perplexity_response(company_name)
                    st.markdown(response)
                st.info(f"현재 사용한 크레딧: ${credit_used:.2f} / ${CREDIT_LIMIT:.2f}")
            else:
                st.warning("크레딧 한도에 도달했습니다. 더 이상 API 호출을 할 수 없습니다.")
        else:
            st.warning("기업명을 입력해주세요.")


if __name__ == "__main__":
    main()