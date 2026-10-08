from pydantic import BaseModel, Field, field_validator


class RequestOtpRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned.count("@") != 1 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned) or cleaned.startswith("@") or cleaned.endswith("@"):
            raise ValueError("Enter a valid email address.")
        local, domain = cleaned.rsplit("@", 1)
        if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
            raise ValueError("Enter a valid email address.")
        return cleaned


class RequestOtpResponse(BaseModel):
    challenge_id: str
    expires_in_seconds: int
    demo_otp: str | None = None
    demo_mode: bool = False


class VerifyOtpRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    otp: str = Field(..., min_length=4, max_length=8)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return RequestOtpRequest.normalize_email(value)

    @field_validator("otp")
    @classmethod
    def normalize_otp(cls, value: str) -> str:
        cleaned = "".join(ch for ch in value.strip() if ch.isdigit())
        if not 4 <= len(cleaned) <= 8:
            raise ValueError("Enter the numeric OTP.")
        return cleaned


class AuthUserResponse(BaseModel):
    id: str
    email: str
    workspace_id: str


class AuthMeResponse(BaseModel):
    authenticated: bool
    user: AuthUserResponse | None = None
