from flask import Flask, request, jsonify
from flask_cors import CORS  # Thêm CORS để tránh vấn đề cross-origin
from search.search import main  # Import hàm main từ search.py

app = Flask(__name__)
CORS(app)  # Cho phép CORS

# Route /ask nhận yêu cầu POST
@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.json.get('question')  # Lấy câu hỏi từ dữ liệu POST

    try:
        result = main(user_input)  # Gọi hàm main từ search.py để xử lý câu hỏi
        if result is None:
            return jsonify({'answer': 'Xin lỗi, tôi không tìm thấy câu trả lời phù hợp.'}), 200
        return jsonify({'answer': result})  # Trả về câu trả lời dưới dạng JSON
    except Exception as e:
        print(f"Error in processing question: {str(e)}")  # In lỗi để debug
        return jsonify({'answer': f'Đã xảy ra lỗi: {str(e)}'}), 500  # Trả về mã lỗi 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 