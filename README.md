# Dont_Buy_It_Helper

UPDATE 02042019

Fixed the bug where half the image would be GREY

Fixed the bug that crashed on extra large input photos

Increased the output image size a bit

The program now waits for image read and write, and seems to be slower

Harley Hou 12/01/2019

项目里的一切我都是第一次用:)

目前只有淘宝好用，其他网站适配请自行解决。
使用方法：点击淘宝商品页面中左上角缩略图下带有数字的方块
商品图片中的人脸会被换成face文件夹中的人脸。
弹出的图片点一下会自动关闭

This program currently works on Taobao.com only.
On clicking the box under thumbnail photo, a popup window should be displayed containing
the photo with all of its human face swapped.
If there is no face in the photo (or no detectable face), original photo should be displayed
The popup window will close on clicking the image itself

运行前需要安装以下软件与库：

You will need: (in windows environment)
The code itself of course
python(3)
openCV and Dlib for face processing
Cmake for Dlib
Visual studio for Cmake (for noobs like me: Do not untick the boxes when installing unless u know what u r doing)
numpy
flask

script.user.js放到你的浏览器里面。需要能运行userscript的插件。如tampermonkey。复制粘贴内容即可。

Install provided userscript (script.user.js) to ur browser (with Chrome: tampermonkey or Firefox: greasemonkey)
I'm using tampermonkey, just paste the content and save works for me.
Should work on firefox but never tried

运行方法：直接在终端中运行flask_test.py即可， 然后就能用了
要更换“脸”的图片的话，把face 文件夹中的图片换掉就行。要重新运行python程序。
增加适配网站需要修改userscript， 可以看看我在script.user.js里改的东西，复制粘贴一下。 只要能拿到原图片的url就行。

To run the code: Download the Flask folder first, goto Flask folder, simply type >python flask_test.py in terminal
To use another face: put another jpg photo containing one face into the face folder and restart the program

业务逻辑（？）：
后端python程序启动时识别face图片并保存计算结果
网页上点击方块后url传入python并下载图片进行替换。
网页获取替换好的图片与新窗口并弹出。
就这样

How it works:
Pretty simple, face image is processed on startup.
Get url with the userscript and POST it to local server. Download the image.
Swap the faces, GET the popup page that displays the image. Popup with the userscript.
