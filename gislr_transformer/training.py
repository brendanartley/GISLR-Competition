import wandb

from gislr_transformer.callbacks import *
from gislr_transformer.models import *
from gislr_transformer.helpers import *

def train(config, CFG):

    # # Set specific GPU for training - For sweeps this is set via bash environment variables
    # set_specific_gpu(ID=config.device)

    # Init wandb/seed
    if config.no_wandb == False:
        wandb.init(project=config.project, config=config)
    set_seeds(seed=config.seed)

    # Get data
    print('-'*15 + " Classifier Training " + "-"*15)
    X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data = load_data(
        val_fold=config.val_fold,
        train_all=config.train_all,
        CFG=CFG,
    )

    # # StatsDict: already computed
    # CFG.statsdict = get_all_stats(X_train=X_train, CFG=CFG)
    # with open('./random.txt', 'w+') as f:
    #     print(CFG.statsdict, file=f)
    CFG.statsdict = {'POSE_MEAN': np.array([[0.4190867 , 0.64570004],[0.47534773, 0.6581575 ],[0.9742574 , 0.8960915 ],[0.76645386, 0.68403614],[0.71533513, 0.61845773],[0.6833763 , 0.5822981 ],[0.68244714, 0.6138692 ],[0.47107282, 0.4247563 ],[0.52839553, 0.37678388],[0.40221563, 0.37658826],[0.57736486, 0.4028722 ],[0.33917505, 0.40404585],[0.36463505, 0.3716686 ],[0.39024717, 0.36453778],[0.42458284, 0.37238878],[0.39266124, 0.3773577 ],[0.5558347 , 0.3711966 ],[0.5311662 , 0.36420116],[0.49782327, 0.37203106],[0.52916056, 0.37686157],[0.46500883, 0.42418867],[0.46440336, 0.43667448],[0.4260663 , 0.43576282],[0.50103295, 0.43517402]], dtype=np.float32), 'POSE_STD': np.array([[0.28084287, 0.07205499],[0.29793707, 0.07083685],[0.10349505, 0.09045887],[0.09346242, 0.12825452],[0.111122  , 0.14830112],[0.1006303 , 0.13695839],[0.09304772, 0.12864468],[0.05871487, 0.0735544 ],[0.0573875 , 0.07377331],[0.06086949, 0.07392994],[0.05353248, 0.07302487],[0.06068742, 0.07405033],[0.06691797, 0.07317372],[0.06596402, 0.07328344],[0.06357788, 0.07250924],[0.06558356, 0.0726615 ],[0.05873729, 0.07321404],[0.06005873, 0.07334504],[0.06096425, 0.07250708],[0.05951796, 0.07251339],[0.06528168, 0.07299399],[0.06330799, 0.07273301],[0.06375591, 0.07248266],[0.06047608, 0.0724477 ]], dtype=np.float32), 'LEFT_HANDS_MEAN': np.array([[0.7619037 , 0.673314  ],[0.7142932 , 0.6334268 ],[0.66480964, 0.5924167 ],[0.62482166, 0.56669873],[0.59701294, 0.551462  ],[0.6729434 , 0.5468663 ],[0.6128007 , 0.51715714],[0.5843762 , 0.5141954 ],[0.5682039 , 0.51344454],[0.68400425, 0.5589306 ],[0.6130567 , 0.53764826],[0.59136665, 0.54462725],[0.583813  , 0.5487585 ],[0.69504803, 0.5801361 ],[0.6283578 , 0.5655684 ],[0.6126555 , 0.5744753 ],[0.6093447 , 0.57885885],[0.70569974, 0.6064784 ],[0.65450865, 0.5957742 ],[0.64142275, 0.5992395 ],[0.637629  , 0.6002077 ]], dtype=np.float32), 'LEFT_HANDS_STD': np.array([[0.09931211, 0.12414423],[0.10399111, 0.12080772],[0.10931057, 0.12359665],[0.11437475, 0.13045996],[0.12332535, 0.13869086],[0.11350865, 0.13243824],[0.12195902, 0.14701143],[0.12539162, 0.15858983],[0.13038012, 0.1690853 ],[0.11385963, 0.14055146],[0.12443443, 0.1577617 ],[0.1258242 , 0.16895783],[0.13116176, 0.17862886],[0.11928932, 0.14852266],[0.13070929, 0.16382858],[0.13078736, 0.17123954],[0.13492964, 0.17813884],[0.12968984, 0.15516372],[0.14124185, 0.16653906],[0.14193699, 0.17222174],[0.14497873, 0.17770725]], dtype=np.float32), 'LIPS_MEAN': np.array([[0.41088957, 0.4787502 ],[0.41503987, 0.47437614],[0.4222455 , 0.46990407],[0.43243885, 0.4648655 ],[0.44827804, 0.46012077],[0.46440032, 0.4619248 ],[0.480219  , 0.45975837],[0.4956243 , 0.4640849 ],[0.5054372 , 0.4689114 ],[0.51219773, 0.4732303 ],[0.5160041 , 0.4775121 ],[0.4157546 , 0.48281825],[0.42276588, 0.48715737],[0.43380147, 0.49219468],[0.448161  , 0.4956314 ],[0.4647738 , 0.4964709 ],[0.48104286, 0.49526376],[0.4948232 , 0.4915382 ],[0.5051664 , 0.48625463],[0.5115963 , 0.48172948],[0.41598755, 0.47833163],[0.4235373 , 0.47659755],[0.43085915, 0.47509408],[0.44017166, 0.47384584],[0.4514634 , 0.4733467 ],[0.4643769 , 0.4734756 ],[0.47697186, 0.4730613 ],[0.4878622 , 0.4732881 ],[0.49673682, 0.47431323],[0.5036243 , 0.47563857],[0.4237506 , 0.47878593],[0.43075344, 0.47894272],[0.44004145, 0.47910693],[0.45133594, 0.4795309 ],[0.46440053, 0.47990987],[0.47715098, 0.47924957],[0.48801422, 0.47857663],[0.49683648, 0.47819328],[0.5034614 , 0.47785515],[0.5109062 , 0.4771972 ]], dtype=np.float32), 'LIPS_STD': np.array([[0.06294865, 0.07283169],[0.06279927, 0.07299843],[0.06277011, 0.07320175],[0.06269374, 0.07342764],[0.06258776, 0.07364003],[0.06192326, 0.07382395],[0.06107807, 0.07350536],[0.05984854, 0.073149  ],[0.05890902, 0.07280571],[0.05807165, 0.07250348],[0.05729685, 0.07225488],[0.06253584, 0.0729985 ],[0.06218699, 0.0733453 ],[0.06169896, 0.07387868],[0.06114373, 0.07429411],[0.06039639, 0.07435507],[0.05942966, 0.07409795],[0.05859166, 0.07351139],[0.05797062, 0.07286145],[0.05754174, 0.07244979],[0.06265942, 0.07287706],[0.06218688, 0.07304429],[0.06205006, 0.07322828],[0.06186248, 0.07343794],[0.06156408, 0.07362124],[0.06109781, 0.07369332],[0.06022214, 0.07349414],[0.0594523 , 0.07319488],[0.05870916, 0.0728907 ],[0.0580617 , 0.07262754],[0.0621339 , 0.07290001],[0.06190608, 0.07302465],[0.06166722, 0.07324412],[0.06133869, 0.07345097],[0.06085313, 0.07353361],[0.0600367 , 0.07330608],[0.05930045, 0.0729788 ],[0.05860867, 0.07266962],[0.05807816, 0.07247863],[0.05746684, 0.07236037]], dtype=np.float32)}

    # Clear all models in GPU
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get Model
    model = get_model(
        num_blocks=config.num_blocks,
        num_heads=config.num_heads, 
        units=config.units,
        landmark_units=config.landmark_units,  
        mlp_dropout_ratio=config.mlp_dropout_ratio, 
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        label_smoothing=config.label_smoothing,
        learning_rate=config.learning_rate,
        clip_norm=config.clip_norm,
        CFG=CFG,
    )

    # Get callbacks
    callbacks = get_callbacks(
        epochs=config.max_epochs,
        warmup_epochs=config.warmup_epochs,
        lr_max=config.learning_rate,
        lr_decay=config.lr_decay,
        num_cycles=config.num_cycles,
        wd_ratio=config.weight_decay,
        do_early_stopping=config.do_early_stopping,
        min_delta=config.min_delta,
        patience=config.patience,
        no_wandb=config.no_wandb,
    )

    # Actual Training
    history=model.fit(
            x=get_train_batch_all_signs(
                X_train, 
                y_train, 
                NON_EMPTY_FRAME_IDXS_TRAIN, 
                n=config.batch_all_signs_n,
                num_classes=config.num_classes,
                CFG=CFG,
                aug=config.aug,
                aug_sampling=config.aug_sampling,
                hand_aug_rotate_ratio=config.hand_aug_rotate_ratio,
                hand_aug_rotate_degrees=config.hand_aug_rotate_degrees,
                hand_aug_expand_ratio=config.hand_aug_expand_ratio,
                hand_aug_expand_pct=config.hand_aug_expand_pct,
                lips_aug_rotate_ratio=config.lips_aug_rotate_ratio,
                lips_aug_rotate_degrees=config.lips_aug_rotate_degrees,
                lips_aug_expand_ratio=config.lips_aug_expand_ratio,
                lips_aug_expand_pct=config.lips_aug_expand_pct,
                face_aug_rotate_ratio=config.face_aug_rotate_ratio,
                face_aug_rotate_degrees=config.face_aug_rotate_degrees,
                eyes_aug_shift_ratio=config.eyes_aug_shift_ratio,
                eyes_aug_shift=config.eyes_aug_shift,
                eyes_aug_expand_ratio=config.eyes_aug_expand_ratio,
                eyes_aug_expand_pct=config.eyes_aug_expand_pct,
                ),
            steps_per_epoch=len(X_train) // (config.num_classes * config.batch_all_signs_n),
            epochs=config.max_epochs,
            # Only used for validation data since training data is a generator
            batch_size=config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=config.verbose,
        )

    # Log results w/ WandB
    log_classification_report(
        model=model,
        history=history,
        validation_data=validation_data,
        num_classes=config.num_classes,
        no_wandb=config.no_wandb,
        CFG=CFG,
    )
    print('Training complete.')