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

    # StatsDict: already computed
    # CFG.statsdict = get_all_stats(X_train=X_train, CFG=CFG)
    CFG.statsdict = {'POSE_MEAN': np.array([[0.46633622, 0.4240536 ],[0.45867133, 0.373784  ],[0.46181643, 0.37736   ],[0.45261687, 0.40010867],[0.45313334, 0.40480456],[0.43614542, 0.64561284],[0.4484901 , 0.6598391 ],[0.9761813 , 0.89589083],[0.76325935, 0.68182313],[0.7099639 , 0.6161903 ],[0.6777221 , 0.5797547 ],[0.67718714, 0.6113952 ],[0.45395005, 0.3722532 ],[0.45475155, 0.3643502 ],[0.45536977, 0.37152776],[0.45473847, 0.37733042],[0.45534915, 0.37017274],[0.45483315, 0.3611874 ],[0.45469826, 0.367665  ],[0.45535198, 0.37427425],[0.4594485 , 0.4237136 ],[0.45882085, 0.43631318],[0.45804474, 0.4359091 ],[0.45781863, 0.4342242 ]], dtype=np.float32), 'POSE_STD': np.array([[0.05858168, 0.07617903],[0.06196702, 0.07624143],[0.10586178, 0.07563455],[0.11064443, 0.07600936],[0.15266573, 0.07561885],[0.28401673, 0.07615038],[0.30110374, 0.07509337],[0.10614169, 0.0933991 ],[0.09435127, 0.13016148],[0.11108568, 0.15089056],[0.10051335, 0.1394862 ],[0.09287425, 0.13098423],[0.14241755, 0.07461724],[0.11975756, 0.07450472],[0.08935333, 0.07406247],[0.11736596, 0.07419308],[0.05070035, 0.07435886],[0.06177724, 0.07509894],[0.0800261 , 0.07499121],[0.06051579, 0.07452413],[0.06548577, 0.07596151],[0.06351445, 0.07586987],[0.09016748, 0.0755309 ],[0.05021851, 0.07555062]], dtype=np.float32), 'LEFT_HANDS_MEAN': np.array([[0.75869346, 0.66998696],[0.7097719 , 0.63058126],[0.65930927, 0.5900772 ],[0.618904  , 0.56465566],[0.59087056, 0.549599  ],[0.66726446, 0.544086  ],[0.6067027 , 0.51483697],[0.57830566, 0.5121041 ],[0.56221086, 0.51145816],[0.6791375 , 0.5559241 ],[0.6078051 , 0.5352497 ],[0.5863684 , 0.54267365],[0.5791031 , 0.5469511 ],[0.69115645, 0.576975  ],[0.62416214, 0.56305933],[0.60877424, 0.5725781 ],[0.6057572 , 0.577219  ],[0.7028756 , 0.603207  ],[0.65143657, 0.59304   ],[0.63851345, 0.5969334 ],[0.63490546, 0.59807736]], dtype=np.float32), 'LEFT_HANDS_STD': np.array([[0.10106158, 0.12571585],[0.10476863, 0.12234896],[0.1093974 , 0.12539777],[0.11426649, 0.13273634],[0.1234008 , 0.14140362],[0.11315253, 0.13402379],[0.12135054, 0.14970353],[0.12469217, 0.16204847],[0.12982582, 0.17308722],[0.1132836 , 0.14246114],[0.12367266, 0.16114126],[0.12492438, 0.17307244],[0.13037677, 0.18315522],[0.11868669, 0.15085693],[0.13010697, 0.16768062],[0.13008882, 0.17569257],[0.1342971 , 0.18288803],[0.12936541, 0.15792547],[0.14111309, 0.17060271],[0.14181338, 0.17685519],[0.14492628, 0.18267772]], dtype=np.float32), 'LIPS_MEAN': np.array([[0.45769352, 0.47939426],[0.45813313, 0.4750107 ],[0.4584668 , 0.47045675],[0.45867002, 0.4652801 ],[0.45874768, 0.4603687 ],[0.45887247, 0.4619916 ],[0.45865375, 0.4595575 ],[0.45827538, 0.46370402],[0.45806175, 0.46836886],[0.45792598, 0.47256923],[0.45801485, 0.47683564],[0.45801198, 0.48348522],[0.458366  , 0.48780724],[0.4587879 , 0.49284703],[0.4591428 , 0.49624422],[0.45937875, 0.4970006 ],[0.4592294 , 0.49564663],[0.458943  , 0.49166977],[0.45856097, 0.48608747],[0.45823926, 0.4812727 ],[0.45769548, 0.47893593],[0.45803356, 0.47716606],[0.4582737 , 0.4756335 ],[0.45852786, 0.47431198],[0.45872262, 0.47371727],[0.45888036, 0.4737115 ],[0.458687  , 0.47312412],[0.45843   , 0.47317424],[0.458201  , 0.47401685],[0.45797396, 0.4752054 ],[0.4580108 , 0.47932437],[0.4582664 , 0.47945398],[0.4585445 , 0.47955698],[0.45877627, 0.47988176],[0.4589286 , 0.48014253],[0.45873004, 0.47932917],[0.4584727 , 0.4784758 ],[0.45822498, 0.47791994],[0.45805368, 0.47744554],[0.45801798, 0.47662103]], dtype=np.float32), 'LIPS_STD': np.array([[0.10080888, 0.07659707],[0.0976407 , 0.07673516],[0.09208792, 0.07690708],[0.08423986, 0.07709396],[0.07280499, 0.07725324],[0.0622867 , 0.07750451],[0.05439064, 0.07712494],[0.05043418, 0.07679018],[0.05054608, 0.07646379],[0.05193912, 0.07615448],[0.05321022, 0.07590489],[0.09655053, 0.07682684],[0.09073339, 0.07725308],[0.08187779, 0.07791283],[0.07127394, 0.07845675],[0.06072269, 0.07858105],[0.05312502, 0.07832868],[0.05021518, 0.07766014],[0.05053236, 0.07683925],[0.05184764, 0.0762542 ],[0.0963515 , 0.07666568],[0.09008428, 0.07687151],[0.08448672, 0.07708108],[0.07751067, 0.07731194],[0.06950647, 0.07751493],[0.06143036, 0.0776047 ],[0.05474281, 0.07737922],[0.05093436, 0.07702701],[0.04955352, 0.07666281],[0.04981965, 0.07635052],[0.08988467, 0.07669736],[0.08436749, 0.07682188],[0.07735248, 0.07704714],[0.06933355, 0.07727893],[0.06117079, 0.07739697],[0.05454649, 0.0771628 ],[0.0508596 , 0.07679975],[0.04956383, 0.07643656],[0.04980446, 0.07619682],[0.05141713, 0.07603587]], dtype=np.float32)}

    # Clear all models in GPU
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get Model
    model = get_model(
        num_blocks=config.num_blocks,
        num_heads=config.num_heads, 
        units=config.units, 
        mlp_dropout_ratio=config.mlp_dropout_ratio, 
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        classifier_drop_rate=config.classifier_drop_rate,
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
                hand_aug_shift_ratio=config.hand_aug_shift_ratio,
                hand_aug_shift=config.hand_aug_shift,
                lips_aug_rotate_ratio=config.lips_aug_rotate_ratio,
                lips_aug_rotate_degrees=config.lips_aug_rotate_degrees,
                lips_aug_expand_ratio=config.lips_aug_expand_ratio,
                lips_aug_expand_pct=config.lips_aug_expand_pct,
                lips_aug_shift_ratio=config.lips_aug_shift_ratio,
                lips_aug_shift=config.lips_aug_shift,
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