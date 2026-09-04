from pydantic import BaseModel, EmailStr, field_validator


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None
    # Customer identity supplied by the system provisioning this account. Both
    # are optional so existing clients are unaffected; when sent, they are what
    # lets an operator recognise the organization without mapping numeric ids
    # by hand. Only applied to a newly created organization.
    organization_display_name: str | None = None
    organization_external_reference: str | None = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str | None
    name: str | None = None
    organization_id: int | None = None
    provider_id: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse
