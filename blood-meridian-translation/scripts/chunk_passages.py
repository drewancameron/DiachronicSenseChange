#!/usr/bin/env python3
"""Generate passage JSON files from raw text chunks."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASSAGES = ROOT / "passages"


def segment_sentences(text: str) -> list[dict]:
    """Split text into sentences with discourse hints."""
    # Split on sentence-ending punctuation, keeping the delimiter
    # Handle dialogue carefully
    raw = re.split(r'(?<=[.!?])\s+', text.strip())

    sentences = []
    for i, s in enumerate(raw):
        s = s.strip()
        if not s:
            continue

        # Determine discourse hint
        hint = "narrative"
        words = s.split()

        # Fragment: very short, no main verb feel
        if len(words) <= 5 and not any(w in s.lower() for w in ['said', 'called', 'cried']):
            hint = "fragment"
        # Dialogue: contains 'said', 'called', etc. or is a direct speech line
        elif any(marker in s.lower() for marker in [', said ', ' said ', ', called ', ' called ', ', cried ']):
            hint = "dialogue"
        elif s.endswith('?') and len(words) <= 10:
            hint = "dialogue"
        # Check for dialect speech patterns (ye, aint, etc.)
        elif any(w in s.lower().split() for w in ["ye", "aint", "dont", "reckon"]) and len(words) < 15:
            hint = "dialogue"

        sentences.append({
            "index": i,
            "text": s,
            "discourse_hint": hint,
        })

    # Re-index
    for i, sent in enumerate(sentences):
        sent["index"] = i

    return sentences


def make_passage(passage_id: str, chapter: str, title: str, text: str,
                 discourse_types: list[str], priority: int, notes: str) -> dict:
    """Create a passage dict in the standard format."""
    sentences = segment_sentences(text)
    return {
        "id": passage_id,
        "chapter": chapter,
        "title": title,
        "source_lines": [0, 0],  # placeholder
        "discourse_types": discourse_types,
        "priority": priority,
        "notes": notes,
        "text": text.strip(),
        "sentences": sentences,
        "sentence_count": len(sentences),
        "glossary_terms_needed": [],
        "translation_status": "pending",
    }


# ═══════════════════════════════════════════════════════════
# CHAPTER II passages
# ═══════════════════════════════════════════════════════════

passages = []

passages.append(make_passage(
    "015_days_of_begging",
    "II",
    "Days of begging and riding",
    "Now come days of begging, days of theft. Days of riding where there rode no soul save he. He's left behind the pinewood country and the evening sun declines before him beyond an endless swale and dark falls here like a thunderclap and a cold wind sets the weeds to gnashing. The night sky lies so sprent with stars that there is scarcely space of black at all and they fall all night in bitter arcs and it is so that their numbers are no less.",
    ["description", "landscape"],
    1,
    "Chapter II opens with lyrical landscape prose. Present tense, catalogue rhythm. Tests poetic register.",
))

passages.append(make_passage(
    "016_prairie_dawn",
    "II",
    "Prairie dawn and the leafhat rider",
    "He keeps from off the king's road for fear of citizenry. The little prairie wolves cry all night and dawn finds him in a grassy draw where he'd gone to hide from the wind. The hobbled mule stands over him and watches the east for light.\n\nThe sun that rises is the color of steel. His mounted shadow falls for miles before him. He wears on his head a hat he's made from leaves and they have dried and cracked in the sun and he looks like a raggedyman wandered from some garden where he'd used to frighten birds.",
    ["description", "landscape"],
    2,
    "Solitary riding. Animal imagery, scarecrow simile. Short declaratives.",
))

passages.append(make_passage(
    "017_hermit_arrival",
    "II",
    "The hermit's hut",
    "Come evening he tracks a spire of smoke rising oblique from among the low hills and before dark he hails up at the doorway of an old anchorite nested away in the sod like a groundsloth. Solitary, half mad, his eyes redrimmed as if locked in their cages with hot wires. But a ponderable body for that. He watched wordless while the kid eased down stiffly from the mule. A rough wind was blowing and his rags flapped about him.\n\nSeen ye smoke, said the kid. Thought you might spare a man a sup of water.\n\nThe old hermit scratched in his filthy hair and looked at the ground. He turned and entered the hut and the kid followed.\n\nInside darkness and a smell of earth. A small fire burned on the dirt floor and the only furnishings were a pile of hides in one corner. The old man shuffled through the gloom, his head bent to clear the low ceiling of woven limbs and mud. He pointed down to where a bucket stood in the dirt. The kid bent and took up the gourd floating there and dipped and drank. The water was salty, sulphurous. He drank on.",
    ["dialogue", "description"],
    2,
    "Arrival at the hermit's dwelling. Dialect dialogue, sensory description. 'Anchorite' — monastic vocabulary.",
))

passages.append(make_passage(
    "018_hermit_well",
    "II",
    "The well and the storm",
    """You reckon I could water my old mule out there?

The old man began to beat his palm with one fist and dart his eyes about.

Be proud to fetch in some fresh. Just tell me where it's at.

What ye aim to water him with?

The kid looked at the bucket and he looked around in the dim hut.

I aint drinkin after no mule, said the hermit.

Have you not got no old bucket nor nothin?

No, cried the hermit. No. I aint. He was clapping the heels of his clenched fists together at his chest.

The kid rose and looked toward the door. Ill find somethin, he said. Where's the well at?

Up the hill, foller the path.

It's nigh too dark to see out here.

It's a deep path. Foller ye feet. Foller ye mule. I caint go.

He stepped out into the wind and looked about for the mule but the mule wasnt there. Far to the south lightning flared soundlessly. He went up the path among the thrashing weeds and found the mule standing at the well.

A hole in the sand with rocks piled about it. A piece of dry hide for a cover and a stone to weight it down. There was a rawhide bucket with a rawhide bail and a rope of greasy leather. The bucket had a rock tied to the bail to help it tip and fill and he lowered it until the rope in his hand went slack while the mule watched over his shoulder.

He drew up three bucketfuls and held them so the mule would not spill them and then he put the cover back over the well and led the mule back down the path to the hut.""",
    ["dialogue", "narrative"],
    2,
    "Extended dialect exchange. Well scene — elemental imagery (water, wind, lightning). Tests vernacular register.",
))

passages.append(make_passage(
    "019_hermit_night",
    "II",
    "The hermit's night sermon",
    """I thank ye for the water, he called.

The hermit appeared darkly in the door. Just stay with me, he said.

That's all right.

Best stay. It's fixin to storm.

You reckon?

I reckon and I reckon right.

Well.

Bring ye bed. Bring ye possibles.

He uncinched and threw down the saddle and hobbled the mule foreleg to rear and took his bedroll in. There was no light save the fire and the old man was squatting by it tailorwise.

Anywheres, anywheres, he said. Where's ye saddle at?

The kid gestured with his chin.

Dont leave it out yonder somethinll eat it. This is a hungry country.

He went out and ran into the mule in the dark. It had been standing looking in at the fire.

Get away, fool, he said. He took up the saddle and went back in.

Now pull that door to fore we blow away, said the old man.

The door was a mass of planks on leather hinges. He dragged it across the dirt and fastened it by its leather latch.

I take it ye lost your way, said the hermit.

No, I went right to it.

He waved quickly with his hand, the old man. No, no, he said. I mean ye was lost to of come here. Was they a sandstorm? Did ye drift off the road in the night? Did thieves beset ye?

The kid pondered this. Yes, he said We got off the road someways or another.

Knowed ye did.

How long you been out here?

Out where?

The kid was sitting on his blanketroll across the fire from the old man. Here, he said. In this place.""",
    ["dialogue", "narrative"],
    2,
    "Domestic scene in the hut. Rapid dialogue, vernacular. The hermit's eccentricity.",
))

passages.append(make_passage(
    "020_hermit_sermon",
    "II",
    "The dried heart and the devil's elbow",
    """The old man didnt answer. He turned his head suddenly aside and seized his nose between his thumb and forefinger and blew twin strings of snot onto the floor and wiped his fingers on the seam of his jeans. I come from Mississippi. I was a slaver, dont care to tell it. Made good money. I never did get caught. Just got sick of it. Sick of niggers. Wait till I show ye somethin.

He turned and rummaged among the hides and handed through the flames a small dark thing. The kid turned it in his hand. Some man's heart, dried and blackened. He passed it back and the old man cradled it in his palm as if he'd weigh it.

They is four things that can destroy the earth, he said. Women, whiskey, money, and niggers.

They sat in silence. The wind moaned in the section of stovepipe that was run through the roof above them to quit the place of smoke. After a while the old man put the heart away.

That thing costed me two hundred dollars, he said.

You give two hundred dollars for it?

I did, for that was the price they put on the black son of a bitch it hung inside of.

He stirred about in the corner and came up with an old dark brass kettle, lifted the cover and poked inside with one finger. The remains of one of the lank prairie hares interred in cold grease and furred with a light blue mold. He clamped the lid back on the kettle and set it in the flames. Aint much but we'll go shares, he said.

I thank ye.

Lost ye way in the dark, said the old man. He stirred the fire, standing slender tusks of bone up out of the ashes.

The kid didnt answer.

The old man swung his head back and forth. The way of the transgressor is hard. God made this world, but he didnt make it to suit everbody, did he?

I dont believe he much had me in mind.

Aye, said the old man. But where does a man come by his notions. What world's he seen that he liked better?

I can think of better places and better ways.

Can ye make it be?

No.

No. It's a mystery. A man's at odds to know his mind cause his mind is aught he has to know it with. He can know his heart, but he dont want to. Rightly so. Best not to look in there. It aint the heart of a creature that is bound in the way that God has set for it. You can find meanness in the least of creatures, but when God made man the devil was at his elbow. A creature that can do anything. Make a machine. And a machine to make the machine. And evil that can run itself a thousand years, no need to tend it. You believe that?

I dont know.

Believe that.""",
    ["dialogue", "philosophical"],
    1,
    "The hermit's theological monologue — 'the devil was at his elbow.' Key thematic passage. Tests philosophical/sermonic register. The dried heart as memento mori.",
))

passages.append(make_passage(
    "021_hermit_departure",
    "II",
    "Thunder and departure",
    """When the old man's mess was warmed he doled it out and they ate in silence. Thunder was moving north and before long it was booming overhead and starting bits of rust in a thin trickle down the stovepipe. They hunkered over their plates and wiped the grease up with their fingers and drank from the gourd.

The kid went out and scoured his cup and plate in the sand and came back banging the tins together as if to fend away some drygulch phantom out there in the dark. Distant thunderheads reared quivering against the electric sky and were sucked away in the blackness again. The old man sat with one ear cocked to the howling waste without. The kid shut the door.

Dont have no bacca with ye do ye?

No I aint, said the kid.

Didnt allow ye did.

You reckon it'll rain?

It's got ever opportunity. Likely it wont.

The kid watched the fire. Already he was nodding. Finally he raised up and shook his head. The hermit watched him over the dying flames. Just go on and fix ye bed, he said.

He did. Spreading his blankets on the packed mud and pulling off his stinking boots. The fluepipe moaned and he heard the mule stamp and snuffle outside and in his sleep he struggled and muttered like a dreaming dog.

He woke sometime in the night with the hut in almost total darkness and the hermit bent over him and all but in his bed.

What do you want? he said. But the hermit crawled away and in the morning when he woke the hut was empty and he got his things and left.""",
    ["narrative", "dialogue"],
    2,
    "Night in the hut, the hermit's sinister advance, dawn departure. Storm imagery. The creeping hermit — menace without explanation.",
))

passages.append(make_passage(
    "022_cattle_drove",
    "II",
    "The cattle drove",
    """All that day he watched to the north a thin line of dust. It seemed not to move at all and it was late evening before he could see that it was headed his way. He passed through a forest of live oak and he watered at a stream and moved on in the dusk and made a fireless camp. Birds woke him where he lay in a dry and dusty wood.

By noon he was on the prairie again and the dust to the north was stretched out along the edge of the earth. By evening the first of a drove of cattle came into view. Rangy vicious beasts with enormous hornspreads. That night he sat in the herders' camp and ate beans and pilotbread and heard of life on the trail.

They were coming down from Abilene, forty days out, headed for the markets in Louisiana. Followed by packs of wolves, coyotes, indians. Cattle groaned about them for miles in the dark.

They asked him no questions, a ragged lot themselves. Crossbreeds some, free niggers, an indian or two.

I had my outfit stole, he said.

They nodded in the firelight.

They got everthing I had. I aint even got a knife.

You might could sign on with us. We lost two men. Turned back to go to Californy.

I'm headed yon way.

I guess you might be goin to Californy ye own self.

I might. I aint decided.

Them boys was with us fell in with a bunch from Arkansas. They was headed down for Bexar. Goin to pull for Mexico and the west.

I'll bet them old boys is in Bexar drinkin they brains out.

I'll bet old Lonnie's done topped ever whore in town.

How far is it to Bexar?

It's about two days.

It's furthern that. More like four I'd say.

How would a man go if he'd a mind to?

You cut straight south you ought to hit the road about half a day.

You going to Bexar?

I might do.

You see old Lonnie down there you tell him get a piece for me. Tell him old Oren. He'll buy ye a drink if he aint blowed all his money in.

In the morning they ate flapjacks with molasses and the herders saddled up and moved on. When he found his mule there was a small fibre bag tied to the animal's rope and inside the bag there was a cupful of dried beans and some peppers and an old greenriver knife with a handle made of string. He saddled up the mule, the mule's back galled and balding, the hooves cracked. The ribs like fishbones. They hobbled on across the endless plain.""",
    ["dialogue", "narrative"],
    2,
    "Campfire with cattle drovers. Rapid vernacular dialogue, multiple speakers. The gift of the knife.",
))

passages.append(make_passage(
    "023_bexar_arrival",
    "II",
    "Arrival at Bexar",
    """He came upon Bexar in the evening of the fourth day and he sat the tattered mule on a low rise and looked down at the town, the quiet adobe houses, the line of green oaks and cottonwoods that marked the course of the river, the plaza filled with wagons with their osnaburg covers and the whitewashed public buildings and the Moorish churchdome rising from the trees and the garrison and the tall stone powderhouse in the distance. A light breeze stirred the fronds of his hat, his matted greasy hair. His eyes lay dark and tunneled in a caved and haunted face and a foul stench rose from the wells of his boot tops. The sun was just down and to the west lay reefs of bloodred clouds up out of which rose little desert nighthawks like fugitives from some great fire at the earth's end. He spat a dry white spit and clumped the cracked wooden stirrups against the mule's ribs and they staggered into motion again.

He went down a narrow sandy road and as he went he met a deadcart bound out with a load of corpses, a small bell tolling the way and a lantern swinging from the gate. Three men sat on the box not unlike the dead themselves or spirit folk so white they were with lime and nearly phosphorescent in the dusk. A pair of horses drew the cart and they went on up the road in a faint miasma of carbolic and passed from sight. He turned and watched them go. The naked feet of the dead jostled stiffly from side to side.""",
    ["description", "landscape"],
    1,
    "Panoramic arrival at Bexar. Long cataloguing sentence, landscape painting. The deadcart — mortality introduced. Key passage for McCarthy's descriptive register.",
))

passages.append(make_passage(
    "024_bexar_night",
    "II",
    "Night in Bexar",
    """It was dark when he entered the town, attended by barking dogs, faces parting the curtains in the lamplit windows. The light clatter of the mule's hooves echoing in the little empty streets. The mule sniffed the air and swung down an alleyway into a square where there stood in the starlight a well, a trough, a hitchingrail. The kid eased himself down and took the bucket from the stone coping and lowered it into the well. A light splash echoed. He drew the bucket, water dripping in the dark. He dipped the gourd and drank and the mule nuzzled his elbow. When he'd done he set the bucket in the street and sat on the coping of the well and watched the mule drink from the bucket.

He went on through the town leading the animal. There was no one about. By and by he entered a plaza and he could hear guitars and a horn. At the far end of the square there were lights from a cafe and laughter and highpitched cries. He led the mule into the square and up the far side past a long portico toward the lights.

There was a team of dancers in the street and they wore gaudy costumes and called out in Spanish. He and the mule stood at the edge of the lights and watched. Old men sat along the tavern wall and children played in the dust. They wore strange costumes all, the men in dark flatcrowned hats, white nightshirts, trousers that buttoned up the outside leg and the girls with garish painted faces and tortoiseshell combs in their blueblack hair. The kid crossed the street with the mule and tied it and entered the cafe. A number of men were standing at the bar and they quit talking when he entered. He crossed the polished clay floor past a sleeping dog that opened one eye and looked at him and he stood at the bar and placed both hands on the tiles. The barman nodded to him. Digame, he said.""",
    ["narrative", "description"],
    2,
    "Nocturnal Bexar. Sensory catalogue, the fiesta, entering the cafe. Spanish dialogue begins.",
))

passages.append(make_passage(
    "025_cafe_broom",
    "II",
    "The broom and the barman",
    """I aint got no money but I need a drink. I'll fetch out the slops or mop the floor or whatever.

The barman looked across the room to where two men were playing dominoes at a table. Abuelito, he said.

The older of the two raised his head.

Que dice el muchacho.

The old man looked at the kid and turned back to his dominoes.

The barman shrugged his shoulders.

The kid turned to the old man. You speak american? he said.

The old man looked up from his play. He regarded the kid without expression.

Tell him I'll work for a drink. I aint got no money.

The old man thrust his chin and made a clucking noise with his tongue.

The kid looked at the barman.

The old man made a fist with the thumb pointing up and the little finger down and tilted his head back and tipped a phantom drink down his throat. Quiere hecharse una copa, he said. Pero no puede pagar.

The men at the bar watched.

The barman looked at the kid.

Quiere trabajo, said the old man. Quien sabe. He turned back to his pieces and made his play without further consultation.

Quieres trabajar, said one of the men at the bar.

They began to laugh.

What are you laughing at? said the boy.

They stopped. Some looked at him, some pursed their mouths or shrugged. The boy turned to the bartender. You got something I could do for a couple of drinks I know damn good and well.

One of the men at the bar said something in Spanish. The boy glared at them. They winked one to the other, they took up their glasses.

He turned to the barman again. His eyes were dark and narrow. Sweep the floor, he said.

The barman blinked.

The kid stepped back and made sweeping motions, a pantomime that bent the drinkers in silent mirth. Sweep, he said, pointing at the floor.

No esta sucio, said the barman.

He swept again. Sweep, goddamnit, he said.

The barman shrugged. He went to the end of the bar and got a broom and brought it back. The boy took it and went on to the back of the room.

A great hall of a place. He swept in the corners where potted trees stood silent in the dark. He swept around the spittoons and he swept around the players at the table and he swept around the dog. He swept along the front of the bar and when he reached the place where the drinkers stood he straightened up and leaned on the broom and looked at them. They conferred silently among themselves and at last one took his glass from the bar and stepped away. The others followed. The kid swept past them to the door.

The dancers had gone, the music. Across the street sat a man on a bench dimly lit in the doorlight from the cafe. The mule stood where he'd tied it. He tapped the broom on the steps and came back in and took the broom to the corner where the barman had gotten it. Then he came to the bar and stood.

The barman ignored him.

The kid rapped with his knuckles.

The barman turned and put one hand on his hip and pursed his lips.

How about that drink now, said the kid.

The barman stood.

The kid made the drinking motions that the old man had made and the barman flapped his towel idly at him.

Andale, he said. He made a shooing motion with the back of his hand.""",
    ["dialogue", "narrative"],
    2,
    "The broom scene — bilingual humiliation and defiance. Extended dialogue with Spanish. Pantomime comedy.",
))

passages.append(make_passage(
    "026_cafe_violence",
    "II",
    "The barman's pistol and the broken bottles",
    """The kid's face clouded. You son of a bitch, he said. He started down the bar. The barman's expression did not change. He brought up from under the bar an oldfashioned military pistol with a flint lock and shoved back the cock with the heel of his hand. A great wooden clicking in the silence. A clicking of glasses all down the bar. Then the scuffling of chairs pushed back by the players at the wall.

The kid froze. Old man, he said.

The old man didnt answer. There was no sound in the cafe. The kid turned to find him with his eyes.

Esta borracho, said the old man.

The boy watched the barman's eyes.

The barman waved the pistol toward the door.

The old man spoke to the room in Spanish. Then he spoke to the barman. Then he put on his hat and went out.

The barman's face drained. When he came around the end of the bar he had laid down the pistol and he was carrying a bung-starter in one hand.

The kid backed to the center of the room and the barman labored over the floor toward him like a man on his way to some chore. He swung twice at the kid and the kid stepped twice to the right. Then he stepped backward. The barman froze. The kid boosted himself lightly over the bar and picked up the pistol. No one moved. He raked the frizzen open against the bartop and dumped the priming out and laid the pistol down again. Then he selected a pair of full bottles from the shelves behind him and came around the end of the bar with one in each hand.

The barman stood in the center of the room. He was breathing heavily and he turned, following the kid's movements. When the kid approached him he raised the bungstarter. The kid crouched lightly with the bottles and feinted and then broke the right one over the man's head. Blood and liquor sprayed and the man's knees buckled and his eyes rolled. The kid had already let go the bottleneck and he pitched the second bottle into his right hand in a roadagent's pass before it even reached the floor and he backhanded the second bottle across the barman's skull and crammed the jagged remnant into his eye as he went down.

The kid looked around the room. Some of those men wore pistols in their belts but none moved. The kid vaulted the bar and took another bottle and tucked it under his arm and walked out the door. The dog was gone. The man on the bench was gone too. He untied the mule and led it across the square.""",
    ["violence", "narrative"],
    2,
    "Cafe violence — the flintlock, the roadagent's pass, the bottle in the eye. Key action sequence. Tests violence register.",
))

passages.append(make_passage(
    "027_ruined_church",
    "II",
    "The ruined church",
    """He woke in the nave of a ruinous church, blinking up at the vaulted ceiling and the tall swagged walls with their faded frescos. The floor of the church was deep in dried guano and the droppings of cattle and sheep. Pigeons flapped through the piers of dusty light and three buzzards hobbled about on the picked bone carcass of some animal dead in the chancel.

His head was in torment and his tongue swollen with thirst. He sat up and looked around him. He'd put the bottle under his saddle and he found it and held it up and shook it and drew the cork and drank. He sat with his eyes closed, the sweat beaded on his forehead. Then he opened his eyes and drank again. The buzzards stepped down one by one and trotted off into the sacristy. After a while he rose and went out to look for the mule.

It was nowhere in sight. The mission occupied eight or ten ares of enclosed land, a barren purlieu that held a few goats and burros. In the mud walls of the enclosure were cribs inhabited by families of squatters and a few cookfires smoked thinly in the sun. He walked around the side of the church and entered the sacristy. Buzzards shuffled off through the chaff and plaster like enormous yardfowl. The domed vaults overhead were clotted with a dark furred mass that shifted and breathed and chittered. In the room was a wooden table with a few clay pots and along the back wall lay the remains of several bodies, one a child. He went on through the sacristy into the church again and got his saddle. He drank the rest of the bottle and he put the saddle on his shoulder and went out.

The facade of the building bore an array of saints in their niches and they had been shot up by American troops trying their rifles, the figures shorn of ears and noses and darkly mottled with leadmarks oxidized upon the stone. The huge carved and paneled doors hung awap on their hinges and a carved stone Virgin held in her arms a headless child. He stood blinking in the noon heat. Then he saw the mule's tracks. They were just the palest disturbance of the dust and they came out of the door of the church and crossed the lot to the gate in the east wall. He hiked the saddle higher onto his shoulder and set out after them.""",
    ["description", "narrative"],
    1,
    "Waking in the ruined mission. Desecrated sacred space — frescos, guano, corpses, shot saints, headless Virgin. Key register test: ecclesiastical vocabulary.",
))

passages.append(make_passage(
    "028_finding_mule",
    "II",
    "The river and the mule recovered",
    """A dog in the shade of the portal rose and lurched sullenly out into the sun until he had passed and then lurched back. He took the road down the hill toward the river, a ragged figure enough. He entered a deep wood of pecan and oak and the road took a rise and he could see the river below him. Blacks were washing a carriage in the ford and he went down the hill and stood at the edge of the water and after a while he called out to them.

They were sopping water over the black lacquerwork and one of them raised up and turned to look at him. The horses stood to their knees in the current.

What? called the black.

Have you seen a mule.

Mule?

I lost a mule. I think he come this way.

The black wiped his face with the back of his arm. Somethin come down the road about a hour back. I think it went down the river yonder. It might of been a mule. It didnt have no tail nor no hair to speak of but it did have long ears.

The other two blacks grinned. The kid looked off down the river. He spat and set off along the path through the willows and swales of grass.

He found it about a hundred yards downriver. It was wet to its belly and it looked up at him and then lowered its head again into the lush river grass. He threw down the saddle and took up the trailing rope and tied the animal to a limb and kicked it halfheartedly. It shifted slightly to the side and continued to graze. He reached atop his head but he had lost the crazy hat somewhere. He made his way down through the trees and stood looking at the cold swirling waters. Then he waded out into the river like some wholly wretched baptismal candidate.""",
    ["dialogue", "narrative"],
    3,
    "Comic exchange about the mule. The baptismal image at the end. Chapter II closes.",
))

# ═══════════════════════════════════════════════════════════
# CHAPTER III passages
# ═══════════════════════════════════════════════════════════

passages.append(make_passage(
    "029_recruiter",
    "III",
    "Captain White's recruiter",
    """He was lying naked under the trees with his rags spread across the limbs above him when another rider going down the river reined up and stopped.

He turned his head. Through the willows he could see the legs of the horse. He rolled over on his stomach.

The man got down and stood beside the horse.

He reached and got his twinehandled knife.

Howdy there, said the rider.

He didnt answer. He moved to the side to see better through, the branches.

Howdy there. Where ye at?

What do you want?

Wanted to talk to ye.

What about?

Hell fire, come on out. I'm white and Christian.

The kid was reaching up through the willows trying to get his breeches. The belt was hanging down and he tugged at it but the breeches were hung on a limb.

Goddamn, said the man. You aint up in the tree are ye?

Why dont you go on and leave me the hell alone.

Just wanted to talk to ye. Didnt intend to get ye all riled up.

You done got me riled.

Was you the feller knocked in that Mexer's head yesterday evenin? I aint the law.

Who wants to know?

Captain White. He wants to sign that feller up to join the army.

The army?

Yessir.

What army?

Company under Captain White. We goin to whip up on the Mexicans.

The war's over.

He says it aint over. Where are you at?

He rose and hauled the breeches down from where he'd hung them and pulled them on. He pulled on his boots and put the knife in the right bootleg and came out from the willows pulling on his shirt.

The man was sitting in the grass with his legs crossed. He was dressed in buckskin and he wore a plug hat of dusty black silk and he had a small Mexican cigar in the corner of his teeth. When he saw what clawed its way out through the willows he shook his head.

Kindly fell on hard times aint ye son? he said.

I just aint fell on no good ones.

You ready to go to Mexico?

I aint lost nothin down there.

It's a chance for ye to raise ye self in the world. You best make a move someway or another fore ye go plumb in under.

What do they give ye?

Ever man gets a horse and his ammunition. I reckon we might find some clothes in your case.

I aint got no rifle.

We'll find ye one.

What about wages?

Hell fire son, you wont need no wages. You get to keep ever-thing you can raise. We goin to Mexico. Spoils of war. Aint a man in the company wont come out a big landowner. How much land you own now?

I dont know nothin about soldierin.

The man eyed him. He took the unlit cigar from his teeth and turned his head and spat and put it back again. Where ye from? he said.

Tennessee.

Tennessee. Well I dont misdoubt but what you can shoot a rifle.

The kid squatted in the grass. He looked at the man's horse. The horse was fitted out in tooled leather with worked silver trim. It had a white blaze on its face and four white stockings and it was cropping up great teethfuls of the rich grass. Where you from, said the kid.

I been in Texas since thirty-eight. If I'd not run up on Captain White I dont know where I'd be this day. I was a sorrier sight even than what you are and he come along and raised me up like Lazarus. Set my feet in the path of righteousness. I'd done took to drinkin and whorin till hell wouldnt have me. He seen somethin in me worth savin and I see it in you. What do ye say?

I dont know.

Just come with me and meet the captain.

The boy pulled at the halms of grass. He looked at the horse again. Well, he said. Dont reckon it'd hurt nothin.""",
    ["dialogue"],
    2,
    "The recruitment scene — extended vernacular dialogue. Lazarus allusion. Tests sustained dialect register.",
))

passages.append(make_passage(
    "030_captain_white",
    "III",
    "Captain White's speech",
    """They rode through the town with the recruiter splendid on the stockingfooted horse and the kid behind him on the mule like something he'd captured. They rode through narrow lanes where the wattled huts steamed in the heat. Grass and prickly pear grew on the roofs and goats walked about on them and somewhere off in that squalid kingdom of mud the sound of the little deathbells tolled thinly. They turned up Commerce Street through the Main Plaza among rafts of wagons and they crossed another plaza where boys were selling grapes and figs from little trundlecarts.

A few bony dogs slank off before them. They rode through the Military Plaza and they passed the little street where the boy and the mule had drunk the night before and there were clusters of women and girls at the well and many shapes of wickercovered clay jars standing about. They passed a little house where women inside were wailing and the little hearsecart stood at the door with the horses patient and motionless in the heat and the flies.

The captain kept quarters in a hotel on a plaza where there were trees and a small green gazebo with benches. An iron gate at the hotel front opened into a passageway with a courtyard at the rear. The walls were whitewashed and set with little ornate colored tiles. The captain's man wore carved boots with tall heels that rang smartly on the tiles and on the stairs ascending from the courtyard to the rooms above. In the courtyard there were green plants growing and they were freshly watered and steaming. The captain's man strode down the long balcony and rapped sharply at the door at the end. A voice said for them to come in.

He sat at a wickerwork desk writing letters, the captain. They stood attending, the captain's man with his black hat in his hands. The captain wrote on nor did he look up. Outside the kid could hear a woman speaking in Spanish. Other than that there was just the scratching of the captain's pen.

When he had done he laid down the pen and looked up. He looked at his man and then he looked at the kid and then he bent his head to read what he'd written. He nodded to himself and dusted the letter with sand from a little onyx box and folded it. Taking a match from a box of them on the desk he lit it and held it to a stick of sealing wax until a small red medallion had pooled onto the paper. He shook out the match, blew briefly at the paper and knuckled the seal with his ring. Then he stood the letter between two books on his desk and leaned back in his chair and looked at the kid again. He nodded gravely. Take seats, he said.""",
    ["narrative", "description", "dialogue"],
    1,
    "Riding through Bexar to Captain White's quarters. Dense descriptive catalogues. The captain's entrance — deliberate ceremony with pen and sealing wax.",
))


# ═══════════════════════════════════════════════════════════
# Write all passage files
# ═══════════════════════════════════════════════════════════

for p in passages:
    path = PASSAGES / f"{p['id']}.json"
    with open(path, "w") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {p['id']}: {p['sentence_count']} sentences, {len(p['text'])} chars")

# Also create draft directories
DRAFTS = ROOT / "drafts"
for p in passages:
    (DRAFTS / p["id"]).mkdir(parents=True, exist_ok=True)

print(f"\n  {len(passages)} passages created (015-030)")
