from PIL import Image, ImageOps
import dialogs
import photos
import requests
import json

url = 'https://app.nanonets.com/api/v2/ImageCategorization/LabelFile/'

#img= photos.pick_image()

asset = photos.pick_asset(assets=photos.get_assets(media_type='image'))

if asset is not None:
	img = asset.get_image()
	img.show()

	data = {
		'file': asset.get_image_data(),
		'modelId': ('', '44ca0490-074f-4dfb-a1eb-52352fd3cd69')
	}

	print('Performing image classification...')

	response = requests.post(
		url,
		auth=requests.auth.HTTPBasicAuth('uNsw14E5ukIix_yXVbXiy7V6D4ZN22Yu', ''),
		files=data)

	#print(response.text)

	responseJson = json.loads(response.text)
	if responseJson['message'] == 'Success':
		preds = responseJson['result'][0]['prediction']
		for pred in preds:
			print('{0} - {1:0.2f}'.format(pred['label'], pred['probability']))
	else:
		print('Something went wrong')

