from ai.validation.medical_image_validator import MedicalImageValidator

validator = MedicalImageValidator()

result = validator.validate(
    r"D:\Pictures\IMG20240809124040.jpg"
)

print(result)