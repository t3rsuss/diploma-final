import sys
import lief

path = sys.argv[1]
binary = lief.parse(path)
if binary is None:
    sys.exit("Не вдалося розібрати файл")

kind = "PE" if isinstance(binary, lief.PE.Binary) else "ELF" if isinstance(binary, lief.ELF.Binary) else "інший"
print("Формат:", kind)
print("Точка входу:", hex(binary.entrypoint))
print("Секції:")
for s in binary.sections:
    print(f"  {s.name:<10} розмір={s.size:<8} ентропія={s.entropy:.2f}")