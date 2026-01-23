from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

class MLNN_separate_a(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type = "gs_separate"
        self.flatten = nn.Flatten()

        # Put angle infromation into a higher dimensional latent space
        self.guide_star_latent_space_encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(32, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )
        self.guide_star_latent_space_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(32, 9),
        )

        # Initial feature extraction
        self.feature_extraction_init = nn.Sequential(
            nn.Linear(self.n_features*3, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(256, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )

        # Two steps for angle and feature information together
        self.combined_step_a = nn.Sequential(
            nn.Linear(512+64, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )

        self.combined_step_b = nn.Sequential(
            nn.Linear(1024+64, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )

        # Output step for estimated modes
        self.estimator_step = nn.Sequential(
            nn.Linear(1024+64, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features),
        )


    def forward(self, x_m, x_gs):
        # x_m = batch["modes"]
        # x_gs = batch["guidestars"]
        x_m = self.flatten(x_m)
        x_gs = self.flatten(x_gs)

        x_gs = self.guide_star_latent_space_encoder(x_gs)

        x_m = self.feature_extraction_init(x_m)
        x = cat([x_m, x_gs], dim=1)
        x = self.combined_step_a(x)
        x = cat([x, x_gs], dim=1)
        x = self.combined_step_b(x)
        x = cat([x, x_gs], dim=1)

        x_m = self.estimator_step(x)
        x_gs = self.guide_star_latent_space_decoder(x_gs)

        return x_m, x_gs #{ "modes": x_m, "guidestars": x_gs }

class MLNN_separate_b(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type = "gs_separate"
        self.flatten = nn.Flatten()

        # Put angle infromation into a higher dimensional latent space
        self.guide_star_latent_space_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
        )
        self.guide_star_latent_space_decoder = nn.Sequential(
            nn.Linear(64, 9),
        )

        # Initial feature extraction
        self.feature_extraction_init = nn.Sequential(
            nn.Linear(self.n_features*3, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
        )

        # Two steps for angle and feature information together
        self.combined_step_a = nn.Sequential(
            nn.Linear(512+64, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
        )

        self.combined_step_b = nn.Sequential(
            nn.Linear(1024+64, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
        )

        # Output step for estimated modes
        self.estimator_step = nn.Sequential(
            nn.Linear(1024+64, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features),
        )

    def forward(self, x_m, x_gs):
        # x_m = batch["modes"]
        # x_gs = batch["guidestars"]
        x_m = self.flatten(x_m)
        x_gs = self.flatten(x_gs)

        x_gs = self.guide_star_latent_space_encoder(x_gs)

        x_m = self.feature_extraction_init(x_m)
        x = cat([x_m, x_gs], dim=1)
        x = self.combined_step_a(x)
        x = cat([x, x_gs], dim=1)
        x = self.combined_step_b(x)
        x = cat([x, x_gs], dim=1)

        x_m = self.estimator_step(x)
        x_gs = self.guide_star_latent_space_decoder(x_gs)

        return x_m, x_gs #{ "modes": x_m, "guidestars": x_gs }

class MLNN_separate_ab(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type = "gs_separate"
        self.flatten = nn.Flatten()

        # Put angle infromation into a higher dimensional latent space
        self.guide_star_latent_space_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
        )
        self.guide_star_latent_space_decoder = nn.Sequential(
            nn.Linear(64, 9),
        )

        # Initial feature extraction
        self.feature_extraction_init = nn.Sequential(
            nn.Linear(self.n_features*3, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(256, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )

        # Two steps for angle and feature information together
        self.combined_step_a = nn.Sequential(
            nn.Linear(512+64, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )

        self.combined_step_b = nn.Sequential(
            nn.Linear(1024+64, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do)
        )

        # Output step for estimated modes
        self.estimator_step = nn.Sequential(
            nn.Linear(1024+64, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features),
        )

    def forward(self, x_m, x_gs):
        # x_m = batch["modes"]
        # x_gs = batch["guidestars"]
        x_m = self.flatten(x_m)
        x_gs = self.flatten(x_gs)

        x_gs = self.guide_star_latent_space_encoder(x_gs)

        x_m = self.feature_extraction_init(x_m)
        x = cat([x_m, x_gs], dim=1)
        x = self.combined_step_a(x)
        x = cat([x, x_gs], dim=1)
        x = self.combined_step_b(x)
        x = cat([x, x_gs], dim=1)

        x_m = self.estimator_step(x)
        x_gs = self.guide_star_latent_space_decoder(x_gs)

        return x_m, x_gs #{ "modes": x_m, "guidestars": x_gs }
