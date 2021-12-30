import qrcode
import sys

# Test input arguments. Requires sys.
if len(sys.argv) > 1:
    data = sys.argv[1]
else:
    sys.exit("Missing argument")

filename = "qrcode."+data+".png"
print("Saving qr code for", data, "as", filename )

qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=10)

qr.add_data(data)
qr.make(fit=True)

img = qr.make_image(fill='black', back_color='white')
img.save(filename)
