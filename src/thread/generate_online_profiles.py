import json
import os
import random
import numpy as np
from src.configs import Config
from src.models.model_factory import get_model
from src.prompts import Prompt, Conversation
from .user_bot_system_prompt_builder import build_writing_style_prompt


def get_usernames(bot):
     
    usernames = []
    
    # uncomment if wish to generate new usernames (generates max 100 usernames due to limits)
    # system_prompt = "Generate 350 random unique Reddit usernames, write them in 1 row, splitting them by comma without numerating them, also do not write '.' at the end. Be unique with usernames, do not repeat them. Do not write 'Sure' or anything related, start with usernames directly."


    # conv_style = Conversation(
    #                         system_prompt=system_prompt,
    #                         prompts=[]
    #                     )

    # conv_style.add_prompt(Prompt(role='user'))
    # output = str(bot.continue_conversation(conv_style))

    # you can use this list of 500 GPT-generated usernames if you have many profiles (>100)
    output = "Moonlit_Dreamer, SolarSurfer, WinterWhisper, CreativeVista, SandDuneDweller, LiquidSunshine, EnigmaBlossom, DigitalPixie, LostAstrologer, EchoShuffle, QuantumNote, PulseMuse, WaveSeeker, NeonNavigator, MajorScribbler, StarlitSage, SilentEmissary, ApexVoyager, ElectricSketch, PhantomHarmony, SilentSerpent, ShadowPirate, GlitterGlacier, ColorCascade, SunKissedMyth, CosmicChroma, ThunderGypsy, RippleRebel, JadeJuggler, FireflyFlirt, DriftwoodDreamer, ZenithZealot, SolarScout, StellarSiren, TwilightTracker, LavenderLegend, VioletVeil, RainbowRover, MaroonMystery, WinterWarlock, OpalescentOracle, LunarLantern, CopperCoyote, WhisperWanderer, TruthTurtle, ZenZephyr, GhostGlider, BlazeButterfly, DuskDancer, IvoryInferno, RoseRider, WildWolf, MysticMagnet, CrystalCoyote, SilverSparrow, CosmicBreadbasket, PixelPegasus, AuroraAvatar, OmegaOtter, HarmonyHawk, WaterWizard, EdgeEagle, LightLynx, DesertDolphin, EmberEagle, WindWalker, QuantumQuill, AmberAndroid, GlyphGoblin, CosmicCuttlefish, OceanOcelot, DreamDervish, StarScimitar, MiracleMagpie, GravityGriffin, JewelJester, SonicSwan, EmeraldElephant, SparkSprite, NightfallNarwhal, CometCuckoo, ThunderThistle, EtherealElephant, WaveWanderer, CrimsonCondor, NebulaNautilus, SpiralSerpent, FlameFlamingo, DreamDingo, WaxWitch, SeashellSprite, TemporalTigress, GarnetGolem, FrostFawn, SkySkunk, WispWeasel, StellarStork, BlazeBison, AuroraArmadillo, BronzeBudgie, SpiralSphinx, GossamerGoat, DiamondDuck, PalettePenguin, EchoElephant, RainRaccoon, SonicSnake, LuminousLizard, WonderIgnite, NebulousStarlight, MoonBeamCatcher, MysticSkyTeller, SilentEchoes, MirthfulQuill, StargazerGlitz, KeysOfAeolus, FruitfulPonder, VerdantMaven, AstuteOracle, BlissfulThrone, SilverMilkyway, DownpourSage, FieryPheonix, MythicCentaur, CelestialBard, CloudyTales, RovingChronicle, WhimsicalPixie, SublimeApotheosis, ElysiumDreamer, CrestfallenHarbinger, CinnamonMonsoon, TangyZephyr, ArtisticOrient, DefiantCamellia, TwilightMaestro, MorningCoffeeSip, SolsticeSomber, DawningCanary, CherryBlush, MelancholyBorealis, LyricalMariner, PlushMarmalade, GlitteryNostalgia, DuneDrifter, InfinitesimalComet, OpulentLagoon, ToffeeGalactic, RhapsodicCaptive, StolenHalos, SheerLuminary, ArcticMirage, AquamarineFables, CrystallineCrescent, VivaciousSpright, GraciousGossamer, FrostedHeirloom, BriskWhirlwind, ObliviousMetropolis, PositivelyGalactic, LunarBlossoms, ProsperousCadenza, HallowedVortex, SandcastlesInAir, CrypticResonance, AmberConstellation, GoldenVisions, FleetingFoxfire, BluebellGalaxy, SquigglyCurls, InkyStardust, HoneycombDandelion, MelodicRaven, QuiverFull, GrittyChrysanthemum, NightCanvas, TropicalSquall, VelvetAshes, GlisteningParade, TwirlingSunbeam, EmeraldAlexandrite, FeatheredScribbles, SolitaryHummingbird, SilentStellar, SaffronEmanation, MandalaPassages, PeriaptChronicles, QuirkyQuasar, PiquantAurora, SparkleChrome, GrisailleWhisper, SassyHelvetica, MidnightEmissary, LavenderSpheres, AmbrosialTyphoon, RainbowChameleon, PistachioFrost, SpicyPeppermint, VelvetMorning, JacarandaJouissance, RosyEuphoria, MosaicTrance, WaterfallMelody, CosmopolitanArcade, CloudberryPromise, PapyrusCharms, DewyDaffodils, UpliftingOrion, CrimsonCadence, WhimsyWarp, RustyGalaxy, MundaneMuse, PalettePrismatic, PeachyKeenJellyBean, CosmicBreadbasket, JellyfishBlitz, StarrySplatter, WhiskWizard, DaydreamDolphin, StardustSparrow, CherryBlossomBandit, LaughingLemur, GlitterGladiator, WoollyWillow, DreamDeer, FantasyFalcon, MosaicMaple, VintageValley, TumbleweedTwister, SeaSerpentShadow, ZephyrZebra, XanaduXylophone, MoonlitMeadow, WalrusWhisper, BoomerangButterfly, QuirkyQuinoa, DandyDaffodil, RetroRaccoon, OpalescentOwl, CascadeCacti, FireflyForest, RibbonRoadrunner, JellybeanJamboree, GossamerGlacier, TeaTimeTiger, SunsetSalamander, ParrotParadise, NostalgicNarwhal, EndlessEcho, WhisperingWillow, SunflowerSymphony, MidnightMandolin, DawnDandelion, StardustSombrero, RainbowRambler, NibblingNutmeg, LuminousLynx, BreakfastBurrito, CandyCoyote, FluffyFlamingo, TangerineTuba, ZephyrZeppelin, LavenderLlama, BlueberryBiscuit, SummerSolstice, BumblebeeBallet, HoneycombHummingbird, WhimsicalWhale, PapillionPancake, SprinklesSprite, QuantumQuokka, FlossyFawn, DandyDumpling, MoonlitMarigold, HuckleberryHarmony, GlimmerGoblin, PistachioPirate, ZanyZinnia, GlitterGiraffe, JubileeJackal, LavenderLantern, ParadiseParakeet, LollipopLeprechaun, CranberryCroissant, MoonbeamMoose, SunlitStarfish, WanderingWolf, TwilightTulip, VelvetVulture, WaffleWalrus, XylophoneXerus, YetiYacht, ZestoZebra, AlmondAardvark, BunnyBoulevard, CroissantCrusader, DonutDolphin, ElephantEcho, FroggyFestival, GorillaGiggles, HummingbirdHalo, InchwormInferno, JackalJubilee, KangarooKaleidoscope, LemonadeLlama, MangoMeerkat, NoodleNightingale, OctopusOverture, PancakePanda, QuokkaQuicksand, RaccoonRainbow, StarlightSalamander, TadpoleTango, UmbrellaUnicorn, VoilaVulture, WombatWonderland, XylophoneXenon, YarnYak, ZucchiniZeppelin, IcyHistorian, WiseMoose, FragrantFox, SilverSolar, GhostlyChef, BraveNomad, SleepyOracle, CuteCentaur, PassionPanda, QuantumQuokka, JellybeanJester, CosmicCarrot, DreamyDingo, NiftyNarwhal, FluffyFlamingo, CosmicBreadbasket, ZenZebra, VintageVixen, MajesticMango, CraftyCricket, PrismaticPirate, VibrantVulture, EarthyElf, GlitterGiraffe, RoyalRaccoon, ShimmerShark, TangyToucan, JollyJaguar, WonderWalrus, LoyalLynx, TwilightTiger, HoneyHedgehog, MysticMoose, NovaNightingale, LilyLlama, DandyDolphin, RodeoRhino, HappyHeron, FrozenFerret, RusticRook, MellowMandrill, LuckyLemur, SereneSeal, NeonNewt, BubblyBison, VelvetViper, CheeryCheetah, OceanOtter, RadiantRaccoon, QuirkyQuail, FeatherFlamingo, TidalTurtle, TurquoiseToucan, BoltBarracuda, BoldBadger, AdorableAardvark, SwankySeahorse, AmbitiousArmadillo, JitteryJellyfish, EnigmaElephant, DuskDuck, StellarStarfish, GlisteningGuppy, GracefulGazelle, SolarSparrow, TropicalTarantula, PixelatedDreamscape, DriftwoodDonkey, PricklyPorcupine, GrittyGiraffe, DaintyDolphin, MajesticMarten, ArcticMirage, OracleOstrich, RainyRaven, SundanceSalamander, TumbleweedTurtle, VermillionVulture, OptimalOctopus, WhimsicalWhale, ZestfulZebra, VersatileViper, WanderlustWalrus, XanaduXerus, AudaciousAlbatross, QuaintQuokka, RoamageRooster, SilkySparrow, ExoticElephant, JubilantJackal, CosmicBreadbasket, CosmicCougar, WhirlwindWolf, NebulaNightingale, VanillaVixen, GildedGoat, DreamyDugong, JovialJay, KiwiKingfisher, UrbanUakari, EnergeticEagle, MysticMongoose, GiddyGator, WhimsicalWombat, JazzyJaguar, DreamyWanderer, MagicRealist, AphoticDepth, EtherealEcho, NebulousNimbus, QuantumQuill, GalacticGusto, AzureAlchemy, ZephyrZenith, SolarSerendipity, CelestialCipher, MysticMatrix, HeliosHaven, LunarLabyrinth, NetherNomad, StellarScript, ZenithZephyr, OblivionOracle, AstralArtisan, NovaNomad, CosmicCalligraphy, OrionOasis, PulsarPioneer, VegaVisionary, CygnusCipher, QuasarQuadrant, FathomFable, NebulaNavigator, PolarisPoet, MeteorMaverick, ZodiacZephyr, GlobularGalaxy, SupernovaScribe, AndromedaArtist, SiriusSoothsayer, ArtemisAstronomer, HyperionHermit, TaurusTraveler, NebulaNavigatrix, AstralEmissary, AquilaAnalyst, GeminiGlobetrotter, CapricornCartographer, SagittariusSage, ScorpioScribe, ProximaPilgrim, VelaVisionary, GalacticGardener, AndromedaAdventurer, PolarisPioneer, CygnusCartographer, SolarSailor, ShimmyShake, FunkyForce, HoppingHegemony, PurePumpernickel, QuackyQueen, RainbowRhinestone, JazzyJamboree, TimelyTeddy, ZestyZephyr, BopsieBunny, SquigglySquid, DoodlesDolphin, NobleNightingale, FantasticallyFlora, JigglyJelly, MysticMacaw, GroovyGoose, FluffyFennec, PristinePegasus, RaspberryRobin, TerrificTurtle, WhimsicalWalrus, BelovedButterfly, KookyKoala, TwinklingTweety, GigglingGecko, ZingyZebra, JauntyJackal, QuirkyQuokka, UpliftingUnicorn, SassyStarfish"

    usernames = output.split(", ")
    usernames = (np.unique(usernames)).tolist()
    print(len(usernames))
    print(usernames)
     
    return usernames


def get_styles(bot, profiles, user_style_prompt):
                    # sample writing styles before running thread using GPT
        
        usernames = get_usernames(bot)

        print(len(profiles.items()))
        
        for pers, profile in profiles.items():
            try:
                style_bot_system_prompt = build_writing_style_prompt(
                        prompt_skeleton=user_style_prompt,
                        profile=profile
                    )
                conv_style = Conversation(
                                system_prompt=style_bot_system_prompt,
                                prompts=[]
                            )
                
                style = bot.continue_conversation(conv_style)
                username = random.choice(usernames)
                usernames.remove(username)
                print(pers, ": username: ", username)
                print(style, '\n')
                profile['style'] = style
                profile['username'] = username
            
            except Exception as e:
                e = e
                print(pers)
                print('\n')
                print(e)

        return profiles


def gen_style_thread(cfg: Config) -> None:

    checker = get_model(cfg.task_config.checker_bot)

    # load the system prompts
    with open(cfg.task_config.user_style_prompt_path, 'r', encoding="utf-8", errors='ignore') as f:
        user_style_prompt = f.read()

    # load the personalities for the user bot
    with open("./data/profiles/user_bot_profiles_300.json", 'r', encoding="utf-8", errors='ignore') as f:
        user_bot_personalities = json.load(f)

    keys = list(user_bot_personalities.keys())
    user_bot_personalities = {key: user_bot_personalities[key] for key in keys}
            

    # generate writing styles + usernames
    online_profiles = get_styles(checker, user_bot_personalities, user_style_prompt)
    output_file = os.path.join('data/profiles', 'user_bot_gen_online_profiles_300.json')

    
    with open(output_file, 'w') as f:
        json.dump(online_profiles, f)        