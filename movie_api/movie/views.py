from rest_framework import viewsets
from .serializer import MovieSerializer
from rest_framework.response import  Response
from .models import Movie
from rest_framework.decorators import api_view

from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import networkx as nx

class MovieViewSet(viewsets.ModelViewSet):
    queryset = Movie.objects.all()
    serializer_class = MovieSerializer

from django.http import HttpResponse, JsonResponse, FileResponse
import json

@api_view(['GET'])
def keyword(request, diary):
    textrank = TextRank(diary)
    keyword_return = []
    for i in (textrank.keywords(word_num=3)):
        keyword_return.append(i)
    return JsonResponse({
	'message' : 'this is keyword',
	'keywords' : keyword_return
	}
    )

from rest_framework.parsers import JSONParser
from google.cloud import language_v1
@api_view(['POST'])
def keyword_abstract(request):
	if request.method == 'POST':
		if request.META['CONTENT_TYPE'] == "application/json":
			request = json.load(request)
			text = request.get('diary','')
			client = language_v1.LanguageServiceClient()
			
			document = language_v1.types.Document(
					content = ''.join(text),
					type_ = language_v1.Document.Type.PLAIN_TEXT,
					language = "ko"
					)

			response = client.analyze_entities(
					document=document,
					encoding_type = 'UTF8',
					)
			return_keyword = []
			for entity in response.entities:
				return_keyword.append(entity.name)
			
			new_list = []

			for v in return_keyword:
				if v not in new_list:
					new_list.append(v)

			if len(new_list) < 3:
				new_list.append("null")
				new_list.append("null")
			return JsonResponse({
				"message" : "This is Keyword",
				"keyword" : new_list[:3],
				},json_dumps_params={'ensure_ascii':False},status=200)
		elif request.META['CONTENT_TYPE'] == "application/x-www-form-urlencoded":
			return Response("x-www-form-urlencoded")
		elif request.META['CONTENT_TYPE'] == "multipart/form-data":
			return Response("form=data")
		elif request.META['CONTENT_TYPE'] == "text/plain":
			return Response("robot")
		else:
			data = JSONParser().parse(request)
			diary = data['diary']
			return Response(diary)
		return Response("not josn")

@api_view(['GET'])
def get_keyword(request, text):
	client = language_v1.LanguageServiceClient()

	document = language_v1.types.Document(
			content = text,
			type_ = language_v1.Document.Type.PLAIN_TEXT,
			language = "ko"
			)

	response = client.analyze_entities(
			document=document,
			encoding_type = 'UTF8',
			)
	return_keyword = []
	for entity in response.entities:
		return_keyword.append(entity.name)
	return JsonResponse({
		"message" : "This is Keyword",
		"keyword" : return_keyword[:3],
		},json_dumps_params={'ensure_ascii':False}, status=200)


#---------------------------------------#

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch import Tensor
import time
import cv2
from multi_key_dict import multi_key_dict

@api_view(['GET'])
def picture(request, keyword_for_picture):
    start = time.time()

    z = Variable(Tensor(np.random.normal(0, 1, (82, 100)) / 4))

    c = multi_key_dict()

    c['항공모함', '전투함'] = 'aircraft carrier.pt'
    c['비행기','항공기', '공항','여행', '해외','해외여행','여객기'] = 'airplane.pt'
    c['알람', '알람시계','자명종'] = 'alarm clock.pt'
    c['구급차','앰뷸런스', '엠뷸런스', '앰뷸란스', '이송','코로나'] = 'ambulance.pt'
    c['천사', '앤젤','요정'] = 'angel.pt'
    c['갈매기','이주', '동물이동','철새','새떼'] = 'animal migration.pt'
    c['개미','벌레','곤충'] = 'ant.pt'
    c['모루'] = 'anvil.pt'
    c['팔','팔뚝'] = 'arm.pt'
    c['아스파라거스'] = 'asparagus.pt'
    c['도끼','흉기','무기'] = 'axe.pt'
    c['가방','베낭','배낭','백팩','책가방'] = 'backpack.pt'
    c['바나나'] = 'banana.pt'
    c['밴드','반창고','흉터','상처'] = 'bandage.pt'
    c['외양간'] = 'barn.pt'
    c['야구공','야구','야구장'] = 'baseball.pt'
    c['야구배트','배트','빠따','야구빠따','몽둥이','방망이','회초리'] = 'baseball bat.pt'
    c['바구니','과일바구니'] = 'basket.pt'
    c['농구공','농구'] = 'basketball.pt'
    c['박쥐','배트맨'] = 'bat.pt'
    c['욕조','욕실'] = 'bathtub.pt'
    c['해변','해수욕장','해변가','해운대','경포대'] = 'beach.pt'
    c['곰','곰돌이'] = 'bear.pt'
    c['산적','수염'] = 'beard.pt'
    c['침대','침실','잠','꿈'] = 'bed.pt'
    c['벌','꿀벌','말벌','장수말벌','일벌','벌침'] = 'bee.pt'
    c['벨트','허리띠'] = 'belt.pt'
    c['벤치','공원'] = 'bench.pt'
    c['두발자전거','세발자전거','자전거','전기자전거','따릉이','카카오바이크'] = 'bicycle.pt'
    c['망원경'] = 'binoculars.pt'
    c['생일','생일케이크','파티','생일파티','축하','기념일'] = 'birthday cake.pt'
    c['블랙베리', '복분자','산딸기'] = 'blackberry.pt'
    c['블루베리','열매'] = 'blueberry.pt'
    c['책','독서','공부','교과서','전공책','글'] = 'book.pt'
    c['부메랑'] = 'boomerang.pt'
    c['병뚜껑','뚜껑'] = 'bottlecap.pt'
    c['나비넥타이','보타이','보우타이','리본'] = 'bowtie.pt'
    c['팔찌','머리끈'] = 'bracelet.pt'
    c['뇌','좌뇌','우뇌','천재','뇌세포'] = 'brain.pt'
    c['빵','뚜레주르','파리바게트'] = 'bread.pt'
    c['대교','한강대교'] = 'bridge.pt'
    c['브로콜리'] = 'broccoli.pt'
    c['빗자루','청소'] = 'broom.pt'
    c['양동이','버킷','아이스버킷챌린지'] = 'bucket.pt'
    c['불도저','스윙스'] = 'bulldozer.pt'
    c['버스','고속버스','일반버스','대중교통','만원버스','시내버스','마을버스'] = 'bus.pt'
    c['부쉬','풀숲','풀더미','부시'] = 'bush.pt'
    c['나비','나방'] = 'butterfly.pt'
    c['선인장'] = 'cactus.pt'
    c['케이크'] = 'cake.pt'
    c['계산기','공학용계산기','전자계산기'] = 'calculator.pt'
    c['달력','일정','캘린더'] = 'calendar.pt'
    c['낙타','사막','중동'] = 'camel.pt'
    c['카메라','디카','사진기','사진'] = 'camera.pt'
    c['카모플라주','군복','밀리터리','군대','국방'] = 'camouflage.pt'
    c['캠프파이어','모닥불','불놀이'] = 'campfire.pt'
    c['양초','캔들','촛불','촛농'] = 'candle.pt'
    c['대포','바주카포','박격포','캐논'] = 'cannon.pt'
    c['카누','조정','나룻배'] = 'canoe.pt'
    c['당근'] = 'carrot.pt'
    c['성','궁전','캐슬'] = 'castle.pt'
    c['고양이','집사','냥이','길냥이','캣타워'] = 'cat.pt'
    c['천장선풍기','천장장식'] = 'ceiling fan.pt'
    c['휴대전화','스마트폰','휴대폰','폰','핸드폰'] = 'cell phone.pt'
    c['첼로','현악기'] = 'cello.pt'
    c['식탁의자','의자','걸상'] = 'chair.pt'
    c['샹들리에'] = 'chandelier.pt'
    c['교회','기독교','성경'] = 'church.pt'
    c['원','동그라미','공','펄'] = 'circle.pt'
    c['클라리넷'] = 'clarinet.pt'
    c['시계','시간'] = 'clock.pt'
    c['구름','파마머리','파마','펌'] = 'cloud.pt'
    c['커피잔','커피','아메리카노','녹차'] = 'coffee cup.pt'
    c['나침반','방향'] = 'compass.pt'
    c['컴퓨터','컴공','과제','밤샘','데스크탑'] = 'computer.pt'
    c['쿠키'] = 'cookie.pt'
    c['쿨러'] = 'cooler.pt'
    c['소파'] = 'couch.pt'
    c['소','젖소','소고기','암소','한우'] = 'cow.pt'
    c['크레파스','크레용'] = 'crayon.pt'
    c['악어','라코스테','크로커다일'] = 'crocodile.pt'
    c['여객선','크루즈'] = 'cruise ship.pt'
    c['컵','물컵'] = 'cup.pt'
    c['다이아몬드','보석'] = 'diamond.pt'
    c['설거지','식기세척기'] = 'dishwasher.pt'
    c['다이빙','다이빙대'] = 'diving board.pt'
    c['개','강아지','푸들','애견','반려동물'] = 'dog.pt'
    c['돌고래','돌핀'] = 'dolphin.pt'
    c['도넛','던컨도너츠','던킨','간식','도넛방석'] = 'donut.pt'
    c['문','방문','입구','출입구'] = 'door.pt'
    c['용','괴물'] = 'dragon.pt'
    c['수납장','서랍장','서랍'] = 'dresser.pt'
    c['드릴','전기드릴'] = 'drill.pt'
    c['드럼','북','북치기'] = 'drums.pt'
    c['오리','북경오리'] = 'duck.pt'
    c['덤벨','아령','헬스','헬창','운동'] = 'dumbbell.pt'
    c['귀'] = 'ear.pt'
    c['팔꿈치','엘보우'] = 'elbow.pt'
    c['코끼리'] = 'elephant.pt'
    c['편지봉투','편지','편지지','메일','메일함','이메일'] = 'envelope.pt'
    c['지우개'] = 'eraser.pt'
    c['얼굴','면상','와꾸','표정'] = 'face.pt'
    c['선풍기','휴대용선풍기'] = 'fan.pt'
    c['새털','깃털','털','구스다운'] = 'feather.pt'
    c['울타리','휀스','담장','펜스'] = 'fence.pt'
    c['손가락'] = 'finger.pt'
    c['소화전'] = 'fire hydrant.pt'
    c['난로','벽난로'] = 'fireplace.pt'
    c['소방차','소방서','소방대원'] = 'firetruck.pt'
    c['물고기','생선','회','어류'] = 'fish.pt'
    c['플라밍고','학','두루미'] = 'flamingo.pt'
    c['후레쉬','후레시','손전등','라이트'] = 'flashlight.pt'
    c['쪼리','플립플랍','샌들'] = 'flip flops.pt'
    c['조명'] = 'floor lamp.pt'
    c['꽃','꽃다발','꽃집'] = 'flower.pt'
    c['유에프오','미확인비행물체','외계인'] = 'flying saucer.pt'
    c['발','발바닥','발등','발뒤꿈치'] = 'foot.pt'
    c['포크','삼지창','포카락'] = 'fork.pt'
    c['개구리','황소개구리','두꺼비'] = 'frog.pt'
    c['후라이팬'] = 'frying pan.pt'
    c['호스','소방호스','정원호스'] = 'garden hose.pt'
    c['정원','꽃밭','가든'] = 'garden.pt'
    c['기린','멀대','이광수','광수'] = 'giraffe.pt'
    c['염소수염','염소'] = 'goatee.pt'
    c['골프','골프장','라운드','필드','골퍼','골프공'] = 'golf club.pt'
    c['포도','청포도','거봉','샤인머스캣','샤인머스켓'] = 'grapes.pt'
    c['풀','잔디','자연'] = 'grass.pt'
    c['기타','우쿠렐레','베이스'] = 'guitar.pt'
    c['햄버거','버거','맘스터치','맥도날드','롯데리아','버거킹','카우버거'] = 'hamburger.pt'
    c['망치','망치질'] = 'hammer.pt'
    c['손','손등','손바닥'] = 'hand.pt'
    c['하프'] = 'harp.pt'
    c['모자'] = 'hat.pt'
    c['헤드폰','헤드셋'] = 'headphones.pt'
    c['고슴도치','도치'] = 'hedgehog.pt'
    c['헬리콥터','헬기'] = 'helicopter.pt'
    c['헬멧','방탄헬멧','보호장비','공사'] = 'helmet.pt'
    c['육각형','육각너트','깨박이'] = 'hexagon.pt'
    c['하키퍽'] = 'hockey puck.pt'
    c['하키채'] = 'hockey stick.pt'
    c['말','망아지','말고기','승마'] = 'horse.pt'
    c['병원','의사','환자','병동'] = 'hospital.pt'
    c['열기구','기구'] = 'hot air balloon.pt'
    c['핫도그','명량핫도그'] = 'hot dog.pt'
    c['온탕','열탕','목욕탕'] = 'hot tub.pt'
    c['초시계','모래시계'] = 'hourglass.pt'
    c['식물','화초'] = 'house plant.pt'
    c['집','본가'] = 'house.pt'
    c['허리케인','태풍','혼돈'] = 'hurricane.pt'
    c['아이스크림','베스킨라빈스'] = 'ice cream.pt'
    c['자켓','재킷','마이','정장'] = 'jacket.pt'
    c['범죄자','감옥','감방','범죄'] = 'jail.pt'
    c['캥거루','호주'] = 'kangaroo.pt'
    c['열쇠','집열쇠','집키'] = 'key.pt'
    c['키보드','샷건'] = 'keyboard.pt'
    c['무릎','니킥'] = 'knee.pt'
    c['칼','과도','식칼','칼질'] = 'knife.pt'
    c['사다리','사다리게임','사다리타기'] = 'ladder.pt'
    c['랜턴','램프'] = 'lantern.pt'
    c['노트북','맥북','그램'] = 'laptop.pt'
    c['깻잎','잎','잎새','잎사귀','낙엽','이파리','나뭇잎'] = 'leaf.pt'
    c['다리','두다리','하반신','하체','하의'] = 'leg.pt'
    c['전구','에디슨','꼬마전구'] = 'light bulb.pt'
    c['라이터'] = 'lighter.pt'
    c['등대'] = 'lighthouse.pt'
    c['번개','천둥','천둥번개'] = 'lighting.pt'
    c['선','직선','일자'] = 'line.pt'
    c['사자','동물의왕국','동물원'] = 'lion.pt'
    c['립스틱','립글로즈','립밤'] = 'lipstick.pt'
    c['랍스터','가재','랍스터구이','랍스타','바닷가재'] = 'lobster.pt'
    c['화이트데이','사탕','롤리팝','막대사탕'] = 'lollipop.pt'
    c['우편함','우체국','우체통'] = 'mailbox.pt'
    c['지도','맵'] = 'map.pt'
    c['보드마카','마카'] = 'marker.pt'
    c['성냥','성냥개비','성냥팔이소녀'] = 'matches.pt'
    c['확성기'] = 'megaphone.pt'
    c['인어','인어공주'] = 'mermaid.pt'
    c['가수','마이크','노래방','코노','코인노래방','노래'] = 'microphone.pt'
    c['전자레인지','전자렌지'] = 'microwave.pt'
    c['원숭이'] = 'monkey.pt'
    c['달','초승달','그믐달','보름달','달밤'] = 'moon.pt'
    c['모기'] = 'mosquito.pt'
    c['오토바이','레이싱'] = 'motorbike.pt'
    c['산','등산','산맥','정상'] = 'mountain.pt'
    c['마우스','무선마우스'] = 'mouse.pt'
    c['콧수염','프링글스','내시수염','면도'] = 'moustache.pt'
    c['입','입술'] = 'mouth.pt'
    c['머그','머그컵','머그잔'] = 'mug.pt'
    c['버섯','독버섯','초코송이'] = 'mushroom.pt'
    c['못','못박기'] = 'nail.pt'
    c['목걸이','쥬얼리'] = 'necklace.pt'
    c['코','콧구멍','콧대','콧물','콧볼','냄새'] = 'nose.pt'
    c['바다','노을','바닷가'] = 'ocean.pt'
    c['팔각형','팔각정'] = 'octagon.pt'
    c['문어','쭈꾸미','주꾸미','낙지','해물'] = 'octopus.pt'
    c['양파'] = 'onion.pt'
    c['오븐구이','오븐'] = 'oven.pt'
    c['부엉이','올빼미'] = 'owl.pt'
    c['페인트통','페인트'] = 'paint can.pt'
    c['페인트붓','페인트칠'] = 'paintbrush.pt'
    c['야자수','제주도','제주','하와이','야자나무','야자'] = 'palm tree.pt'
    c['판다','팬더','판다곰','팬더곰'] = 'panda.pt'
    c['바지','아랫도리','청바지','슬랙스','면바지'] = 'pants.pt'
    c['옷핀','클립'] = 'paper clip.pt'
    c['낙하산','배그','낙하'] = 'parachute.pt'
    c['앵무새'] = 'parrot.pt'
    c['여권'] = 'passport.pt'
    c['땅콩','땅콩버터'] = 'peanut.pt'
    c['배'] = 'pear.pt'
    c['강낭콩','콩'] = 'peas.pt'
    c['연필','필기구','필기','쓰기'] = 'pencil.pt'
    c['펭귄','남극','북극','펭수','핑구'] = 'penguin.pt'
    c['피아노','그랜드피아노','악기'] = 'piano.pt'
    c['픽업트럭'] = 'pickup truck.pt'
    c['액자','사진액자'] = 'picture frame.pt'
    c['돼지','꿀꿀이','먹보','먹방'] = 'pig.pt'
    c['베개','베개싸움'] = 'pillow.pt'
    c['파인애플'] = 'pineapple.pt'
    c['피자','도미노피자','피자스쿨'] = 'pizza.pt'
    c['펜치','뻰찌','뻰치'] = 'pliers.pt'
    c['경찰차','경찰'] = 'police car.pt'
    c['연못'] = 'pond.pt'
    c['풀장','수영장'] = 'pool.pt'
    c['하드','아이스바'] = 'popsicle.pt'
    c['엽서','엽서사진'] = 'postcard.pt'
    c['감자','감자전','찐감자','왕감자'] = 'potato.pt'
    c['콘센트'] = 'power outlet.pt'
    c['지갑','장지갑'] = 'purse.pt'
    c['토끼','산토끼'] = 'rabbit.pt'
    c['라쿤','너구리'] = 'raccoon.pt'
    c['라디오'] = 'radio.pt'
    c['비','우박'] = 'rain.pt'
    c['무지개다리','무지개'] = 'rainbow.pt'
    c['농기구'] = 'rake.pt'
    c['리모컨','에어컨리모컨'] = 'remote control.pt'
    c['코뿔소'] = 'rhinoceros.pt'
    c['따발총','총'] = 'rifle.pt'
    c['강','한강','강물'] = 'river.pt'
    c['롤러코스터','놀이기구','놀이동산','놀이공원'] = 'roller coaster.pt'
    c['롤러스케이트','스케이트','롤러장'] = 'rollerskates.pt'
    c['돛단배'] = 'sailboat.pt'
    c['샌드위치','서브웨이'] = 'sandwich.pt'
    c['톱질','톱','전기톱'] = 'saw.pt'
    c['색소폰'] = 'saxophone.pt'
    c['학교버스','통학버스'] = 'school bus.pt'
    c['가위','부엌가위','가위질'] = 'scissors.pt'
    c['전갈','스콜피온'] = 'scorpion.pt'
    c['드라이버'] = 'screwdriver.pt'
    c['거북이','자라'] = 'sea turtle.pt'
    c['시소','놀이터'] = 'see saw.pt'
    c['상어','샤크','죠스'] = 'shark.pt'
    c['양','양떼','목장','양떼목장'] = 'sheep.pt'
    c['신발','구두','운동화','스니커즈'] = 'shoe.pt'
    c['반바지'] = 'shorts.pt'
    c['삽','야전삽','삽질'] = 'shovel.pt'
    c['싱크대','세면대'] = 'sink.pt'
    c['스케이트보드','보드','롱보드'] = 'skateboard.pt'
    c['해골','해골바가지','유령','공포'] = 'skull.pt'
    c['마천루','롯데타워','부르즈칼리파','초고층','초고층빌딩'] = 'skyscraper.pt'
    c['침낭'] = 'sleeping bag.pt'
    c['미소','웃는표정','스마일','기분'] = 'smiley face.pt'
    c['달팽이','느림보'] = 'snail.pt'
    c['뱀','뱀장어','보아뱀','구렁이'] = 'snake.pt'
    c['스노클링','스노클','잠수'] = 'snorkel.pt'
    c['눈보라','서리'] = 'snowflake.pt'
    c['눈사람','겨울'] = 'snowman.pt'
    c['골','축구공','축구','축구선수'] = 'soccer ball.pt'
    c['양말','발목양말'] = 'sock.pt'
    c['보트','고속보트','수상스키'] = 'speedboat.pt'
    c['거미','스파이더맨'] = 'spider.pt'
    c['스푼','숟가락','원딜','수저'] = 'spoon.pt'
    c['스프레드시트','표','엑셀'] = 'spreadsheet.pt'
    c['네모','화면','사각형'] = 'square.pt'
    c['꼬불꼬불','곡선'] = 'squiggle.pt'
    c['다람쥐','청설모'] = 'squirrel.pt'
    c['비상계단','계단'] = 'stairs.pt'
    c['별','별모양','스타'] = 'star.pt'
    c['스테이크','고기'] = 'steak.pt'
    c['스테레오','카세트'] = 'stereo.pt'
    c['진단','청진기'] = 'stethoscope.pt'
    c['바느질'] = 'stitches.pt'
    c['정지선','정지','그만'] = 'stop sign.pt'
    c['가스렌지','가스레인지','인덕션'] = 'stove.pt'
    c['딸기','스트로베리'] = 'strawberry.pt'
    c['가로등','거리불빛','가로등불빛'] = 'streetlight.pt'
    c['완두콩'] = 'string bean.pt'
    c['잠수함','잠수정'] = 'submarine.pt'
    c['손가방','서류가방'] = 'suitcase.pt'
    c['해','낮','태양','여름'] = 'sun.pt'
    c['백조'] = 'swan.pt'
    c['스웨터','맨투맨','니트','상의','롱슬리브'] = 'sweater.pt'
    c['그네','그네놀이','그네타기'] = 'swing set.pt'
    c['검','대검'] = 'sword.pt'
    c['주사기','주사','주사바늘','간호사','감기'] = 'syringe.pt'
    c['티셔츠','티','반팔티','반팔','반팔티셔츠'] = 't-shirt.pt'
    c['테이블','책상','식탁'] = 'table.pt'
    c['주전자'] = 'teapot.pt'
    c['테디베어','곰인형','인형'] = 'teddy-bear.pt'
    c['벨소리','전화기','집전화','전화'] = 'telephone.pt'
    c['티비','텔레비전','예능','드라마'] = 'television.pt'
    c['테니스라켓','라켓','테니스채'] = 'tennis racquet.pt'
    c['텐트','캠핑'] = 'tent.pt'
    c['만리장성','중국'] = 'The Great Wall of China.pt'
    c['모나리자'] = 'The Mona Lisa.pt'
    c['호랑이','짐승'] = 'tiger.pt'
    c['토스트','토스트기계','토스터기'] = 'toaster.pt'
    c['발가락','발톱'] = 'toe.pt'
    c['화장실','변기'] = 'toilet.pt'
    c['이빨','이','어금니','사랑니','앞니','임플란트','교정','치아'] = 'tooth.pt'
    c['칫솔','양치','양치질'] = 'toothbrush.pt'
    c['치약'] = 'toothpaste.pt'
    c['토네이도','돌풍','강풍'] = 'tornado.pt'
    c['트랙터'] = 'tractor.pt'
    c['신호등','신호','신호위반','초록불'] = 'traffic light.pt'
    c['기차','열차','지하철'] = 'train.pt'
    c['나무'] = 'tree.pt'
    c['삼각형','삼각김밥'] = 'triangle.pt'
    c['트럼본'] = 'trombone.pt'
    c['트럭','화물차','포터'] = 'truck.pt'
    c['트럼펫'] = 'trumpet.pt'
    c['우산','양산'] = 'umbrella.pt'
    c['속옷','팬티'] = 'underwear.pt'
    c['밴','승합차'] = 'van.pt'
    c['화분','꽃병'] = 'vase.pt'
    c['바이올린'] = 'violin.pt'
    c['빨래','세탁','세탁기','드럼세탁기'] = 'washing machine.pt'
    c['수박'] = 'watermelon.pt'
    c['워터파크','미끄럼틀','슬라이드'] = 'waterslide.pt'
    c['고래','고래밥'] = 'whale.pt'
    c['사과'] = 'apple.pt'
    c['비둘기','새','참새'] = 'bird.pt'
    c['자동차','차','자가용','쏘카','드라이브'] = 'car.pt'
    c['게','꽃게','대게','간장게장','양념게장'] = 'crab.pt'
    c['왕관','왕'] = 'crown.pt'
    c['눈','눈알','눈동자','동공','렌즈'] = 'eye.pt'
    c['선글라스','안경'] = 'eyeglasses.pt'
    c['에펠탑','송전탑','기지국'] = 'The Eiffel Tower.pt'
    c['바퀴','수레바퀴'] = 'wheel.pt'
    c['풍차','네덜란드'] = 'windmill.pt'
    c['와인병','와인'] = 'wine bottle.pt'
    c['와인잔'] = 'wine glass.pt'
    c['손목시계'] = 'wristwatch.pt'
    c['요가','스트레칭'] = 'yoga.pt'
    c['얼룩말'] = 'zebra.pt'
    c['지그재그'] = 'zigzag.pt'

    category = {'사과': 'apple.pt',
                '자전거': 'bicycle.pt',
                '새': 'bird.pt',
                '자동차': 'car.pt',
                '게': 'crab.pt',
                '왕관': 'crown.pt',
                '눈': 'eye.pt',
                '안경': 'eyeglasses.pt',
                '포크': 'fork.pt',
                '에펠탑': 'the eiffel tower.pt',
                '바퀴': 'wheel.pt',
                '풍차': 'windmill.pt',
                '와인병': 'wine bottle.pt',
                '와인잔': 'wine glass.pt',
                '손목시계': 'wristwatch.pt',
                '요가': 'yoga.pt',
                '얼룩말': 'zebra.pt'}

    # value = category.get(keyword_for_picture)
    # value = c[keyword_for_picture]
    value = c.get(keyword_for_picture)
    if value is None:
        print('There is no category of ' + keyword_for_picture)
        return JsonResponse({'message' : "No value!"})
    else:
        load_generator = torch.load('data_pt/' + value,
                                    map_location=lambda storage, loc: storage)

        gen_result = load_generator(z).cpu().detach().numpy()
        white_result = 1 - gen_result[0][0];  # 흑백전환

        #plt.imshow(gen_result[0][0], cmap='YlGn')  # Wistia 주황배경/노란선
        #plt.show()
        #matplotlib.image.imsave('result.png', white_result, cmap='gray')
        # img = cv2.imread("result.png")
        resize_img = cv2.resize(white_result, (0, 0), fx=81, fy=81, interpolation=cv2.INTER_LANCZOS4)
        matplotlib.image.imsave('./static/result.png', resize_img, cmap='gray')

        src = cv2.imread("./static/result.png", cv2.IMREAD_COLOR)

        resize_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, resize_img = cv2.threshold(resize_img, 120, 255, cv2.THRESH_BINARY)

        matplotlib.image.imsave('./static/result.png', resize_img, cmap='gray')
        
        src = cv2.imread("./static/result.png", cv2.IMREAD_COLOR)

        resize_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, resize_img = cv2.threshold(resize_img, 120,255, cv2.THRESH_BINARY)

        resize_img = cv2.resize(resize_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        ret, resize_img = cv2.threshold(resize_img, 120, 255, cv2.THRESH_BINARY)

        resize_img = cv2.resize(resize_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        
        matplotlib.image.imsave('./static/result.png', resize_img, cmap='gray')

        resize_img = cv2.imread('./static/result.png', cv2.IMREAD_COLOR)

        for x in range(0,233):
            for y in range(0,233):
                if resize_img[x][y][0] == 255 & resize_img[x][y][1] == 255 & resize_img[x][y][2] == 255:
                    resize_img[x][y] = [214,215,219]

        matplotlib.image.imsave('./static/result.png', resize_img,
                                cmap='gray')
        
        return FileResponse(open('./static/result.png','rb'))

@api_view(['GET'])
def random_picture(request, keyword_for_picture):

    c = multi_key_dict()
    
    c['항공모함', '전투함'] = 'aircraft carrier.npy'
    c['비행기','항공기', '공항','여행', '해외','해외여행','여객기'] = 'airplane.npy'
    c['알람', '알람시계','자명종'] = 'alarm clock.npy'
    c['구급차','앰뷸런스', '엠뷸런스', '앰뷸란스', '이송','코로나'] = 'ambulance.npy'
    c['천사', '앤젤','요정'] = 'angel.npy'
    c['갈매기','이주', '동물이동','철새','새떼'] = 'animal migration.npy'
    c['개미','벌레','곤충'] = 'ant.npy'
    c['모루'] = 'anvil.npy'
    c['팔','팔뚝'] = 'arm.npy'
    c['아스파라거스'] = 'asparagus.npy'
    c['도끼','흉기','무기'] = 'axe.npy'
    c['가방','베낭','배낭','백팩','책가방'] = 'backpack.npy'
    c['바나나'] = 'banana.npy'
    c['밴드','반창고','흉터','상처'] = 'bandage.npy'
    c['외양간'] = 'barn.npy'
    c['야구공','야구','야구장'] = 'baseball.npy'
    c['야구배트','배트','빠따','야구빠따','몽둥이','방망이','회초리'] = 'baseball bat.npy'
    c['바구니','과일바구니'] = 'basket.npy'
    c['농구공','농구'] = 'basketball.npy'
    c['박쥐','배트맨'] = 'bat.npy'
    c['욕조','욕실'] = 'bathtub.npy'
    c['해변','해수욕장','해변가','해운대','경포대'] = 'beach.npy'
    c['곰','곰돌이'] = 'bear.npy'
    c['산적','수염'] = 'beard.npy'
    c['침대','침실','잠','꿈'] = 'bed.npy'
    c['벌','꿀벌','말벌','장수말벌','일벌','벌침'] = 'bee.npy'
    c['벨트','허리띠'] = 'belt.npy'
    c['벤치','공원'] = 'bench.npy'
    c['두발자전거','세발자전거','자전거','전기자전거','따릉이','카카오바이크'] = 'bicycle.npy'
    c['망원경'] = 'binoculars.npy'
    c['생일','생일케이크','파티','생일파티','축하','기념일'] = 'birthday cake.npy'
    c['블랙베리', '복분자','산딸기'] = 'blackberry.npy'
    c['블루베리','열매'] = 'blueberry.npy'
    c['책','독서','공부','교과서','전공책','글'] = 'book.npy'
    c['부메랑'] = 'boomerang.npy'
    c['병뚜껑','뚜껑'] = 'bottlecap.npy'
    c['나비넥타이','보타이','보우타이','리본'] = 'bowtie.npy'
    c['팔찌','머리끈'] = 'bracelet.npy'
    c['뇌','좌뇌','우뇌','천재','뇌세포'] = 'brain.npy'
    c['빵','뚜레주르','파리바게트'] = 'bread.npy'
    c['대교','한강대교'] = 'bridge.npy'
    c['브로콜리'] = 'broccoli.npy'
    c['빗자루','청소'] = 'broom.npy'
    c['양동이','버킷','아이스버킷챌린지'] = 'bucket.npy'
    c['불도저','스윙스'] = 'bulldozer.npy'
    c['버스','고속버스','일반버스','대중교통','만원버스','시내버스','마을버스'] = 'bus.npy'
    c['부쉬','풀숲','풀더미','부시'] = 'bush.npy'
    c['나비','나방'] = 'butterfly.npy'
    c['선인장'] = 'cactus.npy'
    c['케이크'] = 'cake.npy'
    c['계산기','공학용계산기','전자계산기'] = 'calculator.npy'
    c['달력','일정','캘린더'] = 'calendar.npy'
    c['낙타','사막','중동'] = 'camel.npy'
    c['카메라','디카','사진기','사진'] = 'camera.npy'
    c['카모플라주','군복','밀리터리','군대','국방'] = 'camouflage.npy'
    c['캠프파이어','모닥불','불놀이'] = 'campfire.npy'
    c['양초','캔들','촛불','촛농'] = 'candle.npy'
    c['대포','바주카포','박격포','캐논'] = 'cannon.npy'
    c['카누','조정','나룻배'] = 'canoe.npy'
    c['당근'] = 'carrot.npy'
    c['성','궁전','캐슬'] = 'castle.npy'
    c['고양이','집사','냥이','길냥이','캣타워'] = 'cat.npy'
    c['천장선풍기','천장장식'] = 'ceiling fan.npy'
    c['휴대전화','스마트폰','휴대폰','폰','핸드폰'] = 'cell phone.npy'
    c['첼로','현악기'] = 'cello.npy'
    c['식탁의자','의자','걸상'] = 'chair.npy'
    c['샹들리에'] = 'chandelier.npy'
    c['교회','기독교','성경'] = 'church.npy'
    c['원','동그라미','공','펄'] = 'circle.npy'
    c['클라리넷'] = 'clarinet.npy'
    c['시계','시간'] = 'clock.npy'
    c['구름','파마머리','파마','펌'] = 'cloud.npy'
    c['커피잔','커피','아메리카노','녹차'] = 'coffee cup.npy'
    c['나침반','방향'] = 'compass.npy'
    c['컴퓨터','컴공','과제','밤샘','데스크탑'] = 'computer.npy'
    c['쿠키'] = 'cookie.npy'
    c['쿨러'] = 'cooler.npy'
    c['소파'] = 'couch.npy'
    c['소','젖소','소고기','암소','한우'] = 'cow.npy'
    c['크레파스','크레용'] = 'crayon.npy'
    c['악어','라코스테','크로커다일'] = 'crocodile.npy'
    c['여객선','크루즈'] = 'cruise ship.npy'
    c['컵','물컵'] = 'cup.npy'
    c['다이아몬드','보석'] = 'diamond.npy'
    c['설거지','식기세척기'] = 'dishwasher.npy'
    c['다이빙','다이빙대'] = 'diving board.npy'
    c['개','강아지','푸들','애견','반려동물'] = 'dog.npy'
    c['돌고래','돌핀'] = 'dolphin.npy'
    c['도넛','던컨도너츠','던킨','간식','도넛방석'] = 'donut.npy'
    c['문','방문','입구','출입구'] = 'door.npy'
    c['용','괴물'] = 'dragon.npy'
    c['수납장','서랍장','서랍'] = 'dresser.npy'
    c['드릴','전기드릴'] = 'drill.npy'
    c['드럼','북','북치기'] = 'drums.npy'
    c['오리','북경오리'] = 'duck.npy'
    c['덤벨','아령','헬스','헬창','운동'] = 'dumbbell.npy'
    c['귀'] = 'ear.npy'
    c['팔꿈치','엘보우'] = 'elbow.npy'
    c['코끼리'] = 'elephant.npy'
    c['편지봉투','편지','편지지','메일','메일함','이메일'] = 'envelope.npy'
    c['지우개'] = 'eraser.npy'
    c['얼굴','면상','와꾸','표정'] = 'face.npy'
    c['선풍기','휴대용선풍기'] = 'fan.npy'
    c['새털','깃털','털','구스다운'] = 'feather.npy'
    c['울타리','휀스','담장','펜스'] = 'fence.npy'
    c['손가락'] = 'finger.npy'
    c['소화전'] = 'fire hydrant.npy'
    c['난로','벽난로'] = 'fireplace.npy'
    c['소방차','소방서','소방대원'] = 'firetruck.npy'
    c['물고기','생선','회','어류'] = 'fish.npy'
    c['플라밍고','학','두루미'] = 'flamingo.npy'
    c['후레쉬','후레시','손전등','라이트'] = 'flashlight.npy'
    c['쪼리','플립플랍','샌들'] = 'flip flops.npy'
    c['조명'] = 'floor lamp.npy'
    c['꽃','꽃다발','꽃집'] = 'flower.npy'
    c['유에프오','미확인비행물체','외계인'] = 'flying saucer.npy'
    c['발','발바닥','발등','발뒤꿈치'] = 'foot.npy'
    c['포크','삼지창','포카락'] = 'fork.npy'
    c['개구리','황소개구리','두꺼비'] = 'frog.npy'
    c['후라이팬'] = 'frying pan.npy'
    c['호스','소방호스','정원호스'] = 'garden hose.npy'
    c['정원','꽃밭','가든'] = 'garden.npy'
    c['기린','멀대','이광수','광수'] = 'giraffe.npy'
    c['염소수염','염소'] = 'goatee.npy'
    c['골프','골프장','라운드','필드','골퍼','골프공'] = 'golf club.npy'
    c['포도','청포도','거봉','샤인머스캣','샤인머스켓'] = 'grapes.npy'
    c['풀','잔디','자연'] = 'grass.npy'
    c['기타','우쿠렐레','베이스'] = 'guitar.npy'
    c['햄버거','버거','맘스터치','맥도날드','롯데리아','버거킹','카우버거'] = 'hamburger.npy'
    c['망치','망치질'] = 'hammer.npy'
    c['손','손등','손바닥'] = 'hand.npy'
    c['하프'] = 'harp.npy'
    c['모자'] = 'hat.npy'
    c['헤드폰','헤드셋'] = 'headphones.npy'
    c['고슴도치','도치'] = 'hedgehog.npy'
    c['헬리콥터','헬기'] = 'heliconpyer.npy'
    c['헬멧','방탄헬멧','보호장비','공사'] = 'helmet.npy'
    c['육각형','육각너트','깨박이'] = 'hexagon.npy'
    c['하키퍽'] = 'hockey puck.npy'
    c['하키채'] = 'hockey stick.npy'
    c['말','망아지','말고기','승마'] = 'horse.npy'
    c['병원','의사','환자','병동'] = 'hospital.npy'
    c['열기구','기구'] = 'hot air balloon.npy'
    c['핫도그','명량핫도그'] = 'hot dog.npy'
    c['온탕','열탕','목욕탕'] = 'hot tub.npy'
    c['초시계','모래시계'] = 'hourglass.npy'
    c['식물','화초'] = 'house plant.npy'
    c['집','본가'] = 'house.npy'
    c['허리케인','태풍','혼돈'] = 'hurricane.npy'
    c['아이스크림','베스킨라빈스'] = 'ice cream.npy'
    c['자켓','재킷','마이','정장'] = 'jacket.npy'
    c['범죄자','감옥','감방','범죄'] = 'jail.npy'
    c['캥거루','호주'] = 'kangaroo.npy'
    c['열쇠','집열쇠','집키'] = 'key.npy'
    c['키보드','샷건'] = 'keyboard.npy'
    c['무릎','니킥'] = 'knee.npy'
    c['칼','과도','식칼','칼질'] = 'knife.npy'
    c['사다리','사다리게임','사다리타기'] = 'ladder.npy'
    c['랜턴','램프'] = 'lantern.npy'
    c['노트북','맥북','그램'] = 'lanpyop.npy'
    c['깻잎','잎','잎새','잎사귀','낙엽','이파리','나뭇잎'] = 'leaf.npy'
    c['다리','두다리','하반신','하체','하의'] = 'leg.npy'
    c['전구','에디슨','꼬마전구'] = 'light bulb.npy'
    c['라이터'] = 'lighter.npy'
    c['등대'] = 'lighthouse.npy'
    c['번개','천둥','천둥번개'] = 'lighting.npy'
    c['선','직선','일자'] = 'line.npy'
    c['사자','동물의왕국','동물원'] = 'lion.npy'
    c['립스틱','립글로즈','립밤'] = 'lipstick.npy'
    c['랍스터','가재','랍스터구이','랍스타','바닷가재'] = 'lobster.npy'
    c['화이트데이','사탕','롤리팝','막대사탕'] = 'lollipop.npy'
    c['우편함','우체국','우체통'] = 'mailbox.npy'
    c['지도','맵'] = 'map.npy'
    c['보드마카','마카'] = 'marker.npy'
    c['성냥','성냥개비','성냥팔이소녀'] = 'matches.npy'
    c['확성기'] = 'megaphone.npy'
    c['인어','인어공주'] = 'mermaid.npy'
    c['가수','마이크','노래방','코노','코인노래방','노래'] = 'microphone.npy'
    c['전자레인지','전자렌지'] = 'microwave.npy'
    c['원숭이'] = 'monkey.npy'
    c['달','초승달','그믐달','보름달','달밤'] = 'moon.npy'
    c['모기'] = 'mosquito.npy'
    c['오토바이','레이싱'] = 'motorbike.npy'
    c['산','등산','산맥','정상'] = 'mountain.npy'
    c['마우스','무선마우스'] = 'mouse.npy'
    c['콧수염','프링글스','내시수염','면도'] = 'moustache.npy'
    c['입','입술'] = 'mouth.npy'
    c['머그','머그컵','머그잔'] = 'mug.npy'
    c['버섯','독버섯','초코송이'] = 'mushroom.npy'
    c['못','못박기'] = 'nail.npy'
    c['목걸이','쥬얼리'] = 'necklace.npy'
    c['코','콧구멍','콧대','콧물','콧볼','냄새'] = 'nose.npy'
    c['바다','노을','바닷가'] = 'ocean.npy'
    c['팔각형','팔각정'] = 'octagon.npy'
    c['문어','쭈꾸미','주꾸미','낙지','해물'] = 'octopus.npy'
    c['양파'] = 'onion.npy'
    c['오븐구이','오븐'] = 'oven.npy'
    c['부엉이','올빼미'] = 'owl.npy'
    c['페인트통','페인트'] = 'paint can.npy'
    c['페인트붓','페인트칠'] = 'paintbrush.npy'
    c['야자수','제주도','제주','하와이','야자나무','야자'] = 'palm tree.npy'
    c['판다','팬더','판다곰','팬더곰'] = 'panda.npy'
    c['바지','아랫도리','청바지','슬랙스','면바지'] = 'pants.npy'
    c['옷핀','클립'] = 'paper clip.npy'
    c['낙하산','배그','낙하'] = 'parachute.npy'
    c['앵무새'] = 'parrot.npy'
    c['여권'] = 'passport.npy'
    c['땅콩','땅콩버터'] = 'peanut.npy'
    c['배'] = 'pear.npy'
    c['강낭콩','콩'] = 'peas.npy'
    c['연필','필기구','필기','쓰기'] = 'pencil.npy'
    c['펭귄','남극','북극','펭수','핑구'] = 'penguin.npy'
    c['피아노','그랜드피아노','악기'] = 'piano.npy'
    c['픽업트럭'] = 'pickup truck.npy'
    c['액자','사진액자'] = 'picture frame.npy'
    c['돼지','꿀꿀이','먹보','먹방'] = 'pig.npy'
    c['베개','베개싸움'] = 'pillow.npy'
    c['파인애플'] = 'pineapple.npy'
    c['피자','도미노피자','피자스쿨'] = 'pizza.npy'
    c['펜치','뻰찌','뻰치'] = 'pliers.npy'
    c['경찰차','경찰'] = 'police car.npy'
    c['연못'] = 'pond.npy'
    c['풀장','수영장'] = 'pool.npy'
    c['하드','아이스바'] = 'popsicle.npy'
    c['엽서','엽서사진'] = 'postcard.npy'
    c['감자','감자전','찐감자','왕감자'] = 'potato.npy'
    c['콘센트'] = 'power outlet.npy'
    c['지갑','장지갑'] = 'purse.npy'
    c['토끼','산토끼'] = 'rabbit.npy'
    c['라쿤','너구리'] = 'raccoon.npy'
    c['라디오'] = 'radio.npy'
    c['비','우박'] = 'rain.npy'
    c['무지개다리','무지개'] = 'rainbow.npy'
    c['농기구'] = 'rake.npy'
    c['리모컨','에어컨리모컨'] = 'remote control.npy'
    c['코뿔소'] = 'rhinoceros.npy'
    c['따발총','총'] = 'rifle.npy'
    c['강','한강','강물'] = 'river.npy'
    c['롤러코스터','놀이기구','놀이동산','놀이공원'] = 'roller coaster.npy'
    c['롤러스케이트','스케이트','롤러장'] = 'rollerskates.npy'
    c['돛단배'] = 'sailboat.npy'
    c['샌드위치','서브웨이'] = 'sandwich.npy'
    c['톱질','톱','전기톱'] = 'saw.npy'
    c['색소폰'] = 'saxophone.npy'
    c['학교버스','통학버스'] = 'school bus.npy'
    c['가위','부엌가위','가위질'] = 'scissors.npy'
    c['전갈','스콜피온'] = 'scorpion.npy'
    c['드라이버'] = 'screwdriver.npy'
    c['거북이','자라'] = 'sea turtle.npy'
    c['시소','놀이터'] = 'see saw.npy'
    c['상어','샤크','죠스'] = 'shark.npy'
    c['양','양떼','목장','양떼목장'] = 'sheep.npy'
    c['신발','구두','운동화','스니커즈'] = 'shoe.npy'
    c['반바지'] = 'shorts.npy'
    c['삽','야전삽','삽질'] = 'shovel.npy'
    c['싱크대','세면대'] = 'sink.npy'
    c['스케이트보드','보드','롱보드'] = 'skateboard.npy'
    c['해골','해골바가지','유령','공포'] = 'skull.npy'
    c['마천루','롯데타워','부르즈칼리파','초고층','초고층빌딩'] = 'skyscraper.npy'
    c['침낭'] = 'sleeping bag.npy'
    c['미소','웃는표정','스마일','기분'] = 'smiley face.npy'
    c['달팽이','느림보'] = 'snail.npy'
    c['뱀','뱀장어','보아뱀','구렁이'] = 'snake.npy'
    c['스노클링','스노클','잠수'] = 'snorkel.npy'
    c['눈보라','서리'] = 'snowflake.npy'
    c['눈사람','겨울'] = 'snowman.npy'
    c['골','축구공','축구','축구선수'] = 'soccer ball.npy'
    c['양말','발목양말'] = 'sock.npy'
    c['보트','고속보트','수상스키'] = 'speedboat.npy'
    c['거미','스파이더맨'] = 'spider.npy'
    c['스푼','숟가락','원딜','수저'] = 'spoon.npy'
    c['스프레드시트','표','엑셀'] = 'spreadsheet.npy'
    c['네모','화면','사각형'] = 'square.npy'
    c['꼬불꼬불','곡선'] = 'squiggle.npy'
    c['다람쥐','청설모'] = 'squirrel.npy'
    c['비상계단','계단'] = 'stairs.npy'
    c['별','별모양','스타'] = 'star.npy'
    c['스테이크','고기'] = 'steak.npy'
    c['스테레오','카세트'] = 'stereo.npy'
    c['진단','청진기'] = 'stethoscope.npy'
    c['바느질'] = 'stitches.npy'
    c['정지선','정지','그만'] = 'stop sign.npy'
    c['가스렌지','가스레인지','인덕션'] = 'stove.npy'
    c['딸기','스트로베리'] = 'strawberry.npy'
    c['가로등','거리불빛','가로등불빛'] = 'streetlight.npy'
    c['완두콩'] = 'string bean.npy'
    c['잠수함','잠수정'] = 'submarine.npy'
    c['손가방','서류가방'] = 'suitcase.npy'
    c['해','낮','태양','여름'] = 'sun.npy'
    c['백조'] = 'swan.npy'
    c['스웨터','맨투맨','니트','상의','롱슬리브'] = 'sweater.npy'
    c['그네','그네놀이','그네타기'] = 'swing set.npy'
    c['검','대검'] = 'sword.npy'
    c['주사기','주사','주사바늘','간호사','감기'] = 'syringe.npy'
    c['티셔츠','티','반팔티','반팔','반팔티셔츠'] = 't-shirt.npy'
    c['테이블','책상','식탁'] = 'table.npy'
    c['주전자'] = 'teapot.npy'
    c['테디베어','곰인형','인형'] = 'teddy-bear.npy'
    c['벨소리','전화기','집전화','전화'] = 'telephone.npy'
    c['티비','텔레비전','예능','드라마'] = 'television.npy'
    c['테니스라켓','라켓','테니스채'] = 'tennis racquet.npy'
    c['텐트','캠핑'] = 'tent.npy'
    c['만리장성','중국'] = 'The Great Wall of China.npy'
    c['모나리자'] = 'The Mona Lisa.npy'
    c['호랑이','짐승'] = 'tiger.npy'
    c['토스트','토스트기계','토스터기'] = 'toaster.npy'
    c['발가락','발톱'] = 'toe.npy'
    c['화장실','변기'] = 'toilet.npy'
    c['이빨','이','어금니','사랑니','앞니','임플란트','교정','치아'] = 'tooth.npy'
    c['칫솔','양치','양치질'] = 'toothbrush.npy'
    c['치약'] = 'toothpaste.npy'
    c['토네이도','돌풍','강풍'] = 'tornado.npy'
    c['트랙터'] = 'tractor.npy'
    c['신호등','신호','신호위반','초록불'] = 'traffic light.npy'
    c['기차','열차','지하철'] = 'train.npy'
    c['나무'] = 'tree.npy'
    c['삼각형','삼각김밥'] = 'triangle.npy'
    c['트럼본'] = 'trombone.npy'
    c['트럭','화물차','포터'] = 'truck.npy'
    c['트럼펫'] = 'trumpet.npy'
    c['우산','양산'] = 'umbrella.npy'
    c['속옷','팬티'] = 'underwear.npy'
    c['밴','승합차'] = 'van.npy'
    c['화분','꽃병'] = 'vase.npy'
    c['바이올린'] = 'violin.npy'
    c['빨래','세탁','세탁기','드럼세탁기'] = 'washing machine.npy'
    c['수박'] = 'watermelon.npy'
    c['워터파크','미끄럼틀','슬라이드'] = 'waterslide.npy'
    c['고래','고래밥'] = 'whale.npy'
    c['사과'] = 'apple.npy'
    c['비둘기','새','참새'] = 'bird.npy'
    c['자동차','차','자가용','쏘카','드라이브'] = 'car.npy'
    c['게','꽃게','대게','간장게장','양념게장'] = 'crab.npy'
    c['왕관','왕'] = 'crown.npy'
    c['눈','눈알','눈동자','동공','렌즈'] = 'eye.npy'
    c['선글라스','안경'] = 'eyeglasses.npy'
    c['에펠탑','송전탑','기지국'] = 'The Eiffel Tower.npy'
    c['바퀴','수레바퀴'] = 'wheel.npy'
    c['풍차','네덜란드'] = 'windmill.npy'
    c['와인병','와인'] = 'wine bottle.npy'
    c['와인잔'] = 'wine glass.npy'
    c['손목시계'] = 'wristwatch.npy'
    c['요가','스트레칭'] = 'yoga.npy'
    c['얼룩말'] = 'zebra.npy'
    c['지그재그'] = 'zigzag.npy'
    
    category = {'사과': 'apple.npy',
                '자전거': 'bicycle.npy',
                '새': 'bird.npy',
                '자동차': 'car.npy',
                '게': 'crab.npy',
                '왕관': 'crown.npy',
                '눈': 'eye.npy',
                '안경': 'eyeglasses.npy',
                '포크': 'fork.npy',
                '에펠탑': 'the eiffel tower.npy',
                '바퀴': 'wheel.npy',
                '풍차': 'windmill.npy',
                '와인병': 'wine bottle.npy',
                '와인잔': 'wine glass.npy',
                '손목시계': 'wristwatch.npy',
                '요가': 'yoga.npy',
                '얼룩말': 'zebra.npy'}
	
    # value = category.get(keyword_for_picture)
    value = c.get(keyword_for_picture)
    # value = "sliced_npydata/" + value

    if value is None:
        print('There is no category of ' + keyword_for_picture)
        return Response('no Value')
    else:

        import random
        value = "sliced_npydata/" + value
        nparry = np.load(value)
        length = len(nparry)
        random_image = nparry[random.randint(0,length-1)].reshape((28,28))
        resize_img = cv2.resize(random_image, (0, 0), fx=81, fy=81, interpolation=cv2.INTER_LANCZOS4)
        resize_img = cv2.bitwise_not(resize_img)

        matplotlib.image.imsave('./static/random.png', resize_img,
                            cmap='gray')
        src = cv2.imread("./static/random.png",cv2.IMREAD_COLOR)
        ret, resize_img = cv2.threshold(resize_img, 120, 255, cv2.THRESH_BINARY)
        resize_img = cv2.resize(resize_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        ret, resize_img = cv2.threshold(resize_img, 120, 255, cv2.THRESH_BINARY)
        resize_img = cv2.resize(resize_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        matplotlib.image.imsave('./static/random.png', resize_img, cmap='gray')
        
        resize_img = cv2.imread('./static/random.png', cv2.IMREAD_COLOR)
        for x in range(0,204):
            for y in range(0, 204):
                if resize_img[x][y][0] == 255 & resize_img[x][y][1] == 255 & resize_img[x][y][2] == 255:
                    resize_img[x][y] = [214, 215, 219]
        
        matplotlib.image.imsave('./static/random.png', resize_img, cmap='gray')
        return FileResponse(open('./static/random.png','rb'))


from rest_framework import viewsets
from .serializer import MovieSerializer
from rest_framework.response import  Response
