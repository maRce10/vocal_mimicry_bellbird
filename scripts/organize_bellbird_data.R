library(warbleR)

bb_est <- readRDS("~/Dropbox/Projects/vggish_test/bellbird_data/extended_selection_table_high_snr_models_and_allmimetic_sounds.RDS")


head(bb_est)


bb_est



durs <- sapply(attr(bb_est, "wave.objects"), length)

mx_dur <- max(durs)

library(pbapply)
out <- pblapply(1:nrow(bb_est), cl = 4, function(i){
  
  x <- attr(bb_est, "wave.objects")[[i]]
  dur <- length(x)
  
  if (dur < mx_dur)
    x@left <- c(rep(0, floor((mx_dur- dur) / 2)), x@left, rep(0, floor((mx_dur- dur) / 2)))
    
  if (length(x) < mx_dur)
    x@left <- c(x@left, rep(0, mx_dur - length(x)))
  
  writeWave(x, filename = file.path("/home/m/Dropbox/Projects/vggish_test/bellbird_data/audio", paste0(bb_est$sound.files[i], ".wav")))

  return(NULL)
  })


sfi <- info_sound_files("/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/raw/bellbird_data/audio")

summary(sfi)

table(sfi$samples)


library(caret)


bb_est$org.sound.files <- sapply(strsplit(bb_est$sound.files,
                                            ".wav", fixed = TRUE), "[", 1)


model_sounds <- as.data.frame(bb_est[grep("babbling", bb_est$species.vocalization,
                                  invert = TRUE), ])

model_sounds$filename <- paste0(model_sounds$sound.files, ".wav")

# model_sounds$species.vocalization[grep("mimetic", model_sounds$species.vocalization)] <- "Mimetic"

model_sounds$species.vocalization <- gsub("-model", "", model_sounds$species.vocalization)

model_sounds <- model_sounds[grep("babbling", model_sounds$species.vocalization,
                                  invert = TRUE), ]

mimetic_sounds <- model_sounds[grep("mimetic", model_sounds$species.vocalization), ]

mimetic_sounds$species.vocalization[grep("adult|aggresive|whistle", mimetic_sounds$species.vocalization)] <- "P.tricarunculatus-Talamanca"

mimetic_sounds$species.vocalization <- gsub(pattern = "mimetic_P.tricarunculatus-", "", mimetic_sounds$species.vocalization)

model_sounds <- model_sounds[grep("mimetic", model_sounds$species.vocalization, invert = TRUE), ]

model_sounds$categorynd <- model_sounds$species.vocalization
mimetic_sounds$categorynd <- mimetic_sounds$species.vocalization

model_sounds$categorynd[grep("P.tricarunculatus", model_sounds$categorynd)] <- "P.tricarunculatus"
mimetic_sounds$categorynd[grep("P.tricarunculatus", mimetic_sounds$categorynd)] <- "P.tricarunculatus"
aggregate(org.sound.files ~ species.vocalization, model_sounds, function(x) length(unique(x)))

set.seed(123)
part <- caret::createDataPartition(y = model_sounds$species.vocalization[model_sounds$species.vocalization != "Mimetic"], times = 1, p  = 0.75)

length(part$Resample1) / nrow(model_sounds)

model_sounds$fold <- "test"
model_sounds$fold[part$Resample1] <- "train"
mimetic_sounds$fold <- "mimetic"

# collapsing dialects
set.seed(123)
partnd <- caret::createDataPartition(y = model_sounds$categorynd, times = 1, p  = 0.75)

length(partnd$Resample1) / nrow(model_sounds)

model_sounds$foldnd <- "test"
model_sounds$foldnd[part$Resample1] <- "train"
mimetic_sounds$foldnd <- "mimetic"

full_dat <- rbind(model_sounds, mimetic_sounds)

full_dat$category <- full_dat$species.vocalization

head(full_dat)
write.csv(full_dat[, c("filename", "fold", "category", "foldnd", "categorynd")], "/home/m/Dropbox/Projects/vggish_test/bellbird_data/bellbird.csv")

nrow(full_dat)

table(full_dat$category[full_dat$fold == "mimetic"])
table(full_dat$categorynd[full_dat$foldnd == "mimetic"])
table(full_dat$category[full_dat$fold != "mimetic"])
table(full_dat$categorynd[full_dat$foldnd != "mimetic"])


