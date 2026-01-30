"""
Prompt templates for LLM question generation.
"""
from typing import Final

SYSTEM_MESSAGE: Final[str] = (
    "Bạn là chuyên gia tạo câu hỏi đánh giá. Luôn trả về JSON hợp lệ."
)

USER_PROMPT_TEMPLATE: Final[str] = """
Bạn đang làm việc với MỘT TÀI LIỆU CỤ THỂ.

METADATA TÀI LIỆU (cực kỳ quan trọng):
- Tên file: "{document_name}"
- Lưu ý: Tên file phản ánh LOẠI TÀI LIỆU, PHÒNG BAN và NGỮ CẢNH SỬ DỤNG
- Bạn PHẢI dựa vào metadata này để hiểu đúng mục đích và phạm vi tài liệu

CÁCH DIỄN GIẢI TÊN FILE:
- Nếu tên file chứa các từ như:
  • "QuyTrinh", "Procedure" → Đây là TÀI LIỆU QUY TRÌNH
  • "ChinhSach", "Policy" → Đây là TÀI LIỆU CHÍNH SÁCH
  • "HDSD", "Guide" → Đây là TÀI LIỆU HƯỚNG DẪN SỬ DỤNG
  • "BaoCao", "Report" → Đây là TÀI LIỆU BÁO CÁO, TỔNG KẾT SỐ LIỆU
  • "DaoTao", "Training" → Đây là TÀI LIỆU GIÁO TRÌNH, BÀI GIẢNG
  • "BieuMau", "Form" → Đây là TÀI LIỆU TEMPLATE MẪU, FORM ĐIỀN THÔNG TIN
  • "DanhSach", "List" → Đây là TÀI LIỆU DANH SÁCH
  • "DuAn", "Project" → Đây là TÀI LIỆU LIÊN QUAN ĐẾN DỰ ÁN
  • "GioiThieu", "Intro" → Đây là TÀI LIỆU GIỚI THIỆU, TỔNG QUAN
  • "ThongBao", "Notice" → Đây là TÀI LIỆU THÔNG BÁO
  • "HR" → Thuộc phòng NHÂN SỰ
  • "IC" → Thuộc phòng TRUYỀN THÔNG NỘI BỘ
  • "BD" → Thuộc phòng PHÁT TRIỂN KINH DOANH
  • "CS" → Thuộc phòng CHĂM SÓC KHÁCH HÀNG
  • "PRT" → Thuộc phòng PARTNERSHIP

TỪ METADATA, HÃY XÁC ĐỊNH:
- Loại tài liệu
- Mục tiêu sử dụng
- Đối tượng đọc chính

NHIỆM VỤ:
Dựa trên {document_name} và nội dung tài liệu:
1. Xác định tài liệu thuộc MỘT trong các loại sau:
   QuyTrinh, ChinhSach, HDSD, BaoCao, BieuMau,
   ThongBao, DanhSach, DuAn, GioiThieu, DaoTao
2. Áp dụng BỘ NGUYÊN TẮC TƯƠNG ỨNG với loại tài liệu đó
3. Chỉ tạo câu hỏi phù hợp với loại tài liệu đã xác định

NGUYÊN TẮC CHUNG (áp dụng cho mọi tài liệu):
1. Chỉ tạo câu hỏi dựa trên nội dung thực tế của tài liệu
2. Không suy diễn, không bổ sung kiến thức bên ngoài
3. Mỗi câu hỏi phải có câu trả lời rõ ràng trong tài liệu
4. Câu hỏi phải phản ánh đúng MỤC ĐÍCH SỬ DỤNG của tài liệu

NGUYÊN TẮC RIÊNG THEO LOẠI TÀI LIỆU:
1. Nếu tài liệu là QuyTrinh:
- Ưu tiên câu hỏi về:
  • Trình tự các bước thực hiện
  • Vai trò và trách nhiệm của từng bên
  • Thẩm quyền phê duyệt / ra quyết định
  • SLA, deadline, mốc thời gian
  • Điều kiện bắt đầu / kết thúc
  • Ngoại lệ và rủi ro
- KHÔNG tạo câu hỏi lý thuyết hoặc khái niệm chung
2. Nếu tài liệu là ChinhSach:
- Ưu tiên câu hỏi về:
  • Phạm vi áp dụng
  • Nguyên tắc, quy định bắt buộc
  • Quyền lợi và nghĩa vụ
  • Điều bị cấm / hạn chế
  • Trách nhiệm và chế tài (nếu có)
- KHÔNG hỏi trình tự thao tác chi tiết
3. Nếu tài liệu là HDSD:
- Ưu tiên câu hỏi về:
  • Cách thực hiện thao tác
  • Các bước sử dụng chức năng
  • Điều kiện trước khi thực hiện
  • Lỗi thường gặp và cách xử lý
- Có thể hỏi “Làm thế nào”, “Thực hiện ra sao”
4. Nếu tài liệu là BaoCao:
- Ưu tiên câu hỏi về:
  • Chỉ số, số liệu chính
  • So sánh giữa các kỳ
  • Xu hướng, tăng/giảm
  • Kết luận hoặc nhận định được nêu trong báo cáo
- KHÔNG hỏi quy trình hay hướng dẫn thao tác
5. Nếu tài liệu là BieuMau:
- Ưu tiên câu hỏi về:
  • Mục đích của biểu mẫu
  • Các trường thông tin cần điền
  • Điều kiện sử dụng biểu mẫu
  • Trường hợp cần / không cần sử dụng
- KHÔNG hỏi lý do nghiệp vụ sâu
6. Nếu tài liệu là ThongBao:
- Ưu tiên câu hỏi về:
  • Nội dung thông báo chính
  • Thời gian hiệu lực
  • Đối tượng áp dụng
  • Hành động cần thực hiện (nếu có)
- KHÔNG hỏi phân tích hoặc so sánh
7. Nếu tài liệu là DanhSach:
- Ưu tiên câu hỏi về:
  • Danh sách gồm những mục nào
  • Số lượng phần tử
  • Thống kê số lượng theo nhóm (nếu có)
- KHÔNG hỏi quy trình hay lý do nghiệp vụ
8. Nếu tài liệu là DuAn:
- Ưu tiên câu hỏi về:
  • Mục tiêu và phạm vi dự án
  • Các mốc thời gian chính
  • Vai trò và trách nhiệm
  • Deliverables / kết quả mong đợi
  • Rủi ro hoặc giả định (nếu có)
9. Nếu tài liệu là GioiThieu:
- Ưu tiên câu hỏi về:
  • Sản phẩm / dịch vụ / công ty là gì
  • Giá trị cốt lõi
  • Đối tượng khách hàng
  • Điểm nổi bật được nêu trong tài liệu
- KHÔNG hỏi chi tiết vận hành
10. Nếu tài liệu là DaoTao:
- Ưu tiên câu hỏi về:
  • Mục tiêu học tập
  • Nội dung chính của bài học
  • Kiến thức / kỹ năng được đào tạo
  • Ví dụ hoặc tình huống được nêu
- Có thể hỏi kiểm tra hiểu bài


YÊU CẦU NGÔN NGỮ (BẮT BUỘC - CỰC KỲ QUAN TRỌNG):
- TẤT CẢ CÁC CÂU HỎI PHẢI ĐƯỢC VIẾT HOÀN TOÀN BẰNG TIẾNG VIỆT
- TUYỆT ĐỐI KHÔNG được sử dụng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác
- Ngay cả khi tài liệu có nội dung tiếng Anh, câu hỏi vẫn PHẢI bằng tiếng Việt
- Ví dụ SAI: "How many brands are there?"
- Ví dụ ĐÚNG: "Có bao nhiêu thương hiệu trong danh sách?"
- Nếu một câu hỏi không thể viết bằng tiếng Việt → KHÔNG tạo câu hỏi đó

YÊU CẦU ĐẦU RA:
- answer_location PHẢI là bản sao NGUYÊN VĂN từ tài liệu
- Không chỉnh sửa, không diễn giải

YÊU CẦU TÍNH DUY NHẤT (CRITICAL):
- MỖI câu hỏi PHẢI hoàn toàn KHÁC BIỆT về ý nghĩa
- KHÔNG được tạo câu hỏi trùng lặp hoặc tương tự nhau
- Kiểm tra kỹ: 2 câu hỏi không được hỏi về CÙNG MỘT thông tin

YÊU CẦU SỐ LƯỢNG (BẮT BUỘC):
- PHẢI tạo CHÍNH XÁC {num_questions} câu hỏi
- Mỗi câu hỏi PHẢI có answer_location tồn tại NGUYÊN VĂN trong tài liệu
- KHÔNG trả ít hơn {num_questions}
- KHÔNG tạo câu hỏi nếu không tìm được câu trả lời rõ ràng
- Nếu một câu hỏi không chắc chắn → chọn câu khác

TÀI LIỆU:
---
{document_content}
---

OUTPUT:
[
  {{
    "question": "...",
    "type": "what|why|when|how|who|list|condition",
    "answer_location": "Trích dẫn nguyên văn đoạn tài liệu chứa câu trả lời"
  }}
]

CHỈ trả về JSON.
NHẮC LẠI: MỌI CÂU HỎI PHẢI BẰNG TIẾNG VIỆT.
"""


def build_user_prompt(
    document_content: str, document_name: str, num_questions: int
) -> str:
    """Fill the prompt template with document metadata and content."""

    return USER_PROMPT_TEMPLATE.format(
        document_name=document_name,
        num_questions=num_questions,
        document_content=document_content,
    )
