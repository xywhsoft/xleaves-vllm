"""Deterministic visual fixtures for image admission and OCR regression checks."""
import base64
import io

from PIL import Image, ImageDraw, ImageFont


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def checks(base, key, http):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 52)
    ocr = Image.new('RGB', (640, 240), 'white')
    ImageDraw.Draw(ocr).text((25, 75), 'ORION 7294', font=font, fill='black')
    count = Image.new('RGB', (480, 240), 'white')
    drawer = ImageDraw.Draw(count)
    for x in [70, 230, 390]:
        drawer.ellipse((x - 40, 80, x + 40, 160), fill='red')
    fixtures = [
        ('ocr', ocr, 'Read the printed text. Reply only with the text.',
         lambda answer: 'ORION' in answer.upper() and '7294' in answer),
        ('count', count, 'How many red circles are shown? Reply only with the number.',
         lambda answer: '3' in answer or 'three' in answer.lower()),
    ]
    rows = []
    for name, image, prompt, expected in fixtures:
        body = dict(model='ornith-1.5-35b', temperature=0, max_tokens=128,
                    chat_template_kwargs={'enable_thinking': False},
                    messages=[dict(role='user', content=[
                        dict(type='image_url', image_url=dict(url=image_url(image))),
                        dict(type='text', text=prompt)])])
        response = http(base, '/v1/chat/completions', key, body, timeout=300)
        message = response['choices'][0]['message']
        answer = message.get('content') or ''
        rows.append(dict(fixture=name, answer=answer, passed=expected(answer),
                         reasoning=message.get('reasoning_content'), usage=response.get('usage')))
    body = dict(model='ornith-1.5-35b', temperature=0, max_tokens=128,
                chat_template_kwargs={'enable_thinking': False},
                messages=[dict(role='user', content=[
                    dict(type='image_url', image_url=dict(url=image_url(Image.new('RGB', (128, 128), 'red')))),
                    dict(type='image_url', image_url=dict(url=image_url(Image.new('RGB', (128, 128), 'blue')))),
                    dict(type='text', text='Name the dominant color of each image, first image then second image. Reply only with two English color names.')])])
    response = http(base, '/v1/chat/completions', key, body, timeout=300)
    message = response['choices'][0]['message']
    answer = message.get('content') or ''
    lowered = answer.lower()
    rows.append(dict(fixture='two_images_ordered_colors', answer=answer,
        passed='red' in lowered and 'blue' in lowered and lowered.index('red') < lowered.index('blue'),
        reasoning=message.get('reasoning_content'), usage=response.get('usage')))
    return rows
