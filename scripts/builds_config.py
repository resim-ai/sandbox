from pydantic import BaseModel


class ResimAppConfig(BaseModel):
    client_id: str
    auth_url: str
    api_url: str


class RegistryAuth(BaseModel):
    profile: str


class ImageRegistry(BaseModel):
    account_id: str
    region: str
    auth: RegistryAuth


class ImageRepo(BaseModel):
    name: str
    registry: str


class BuildCommand(BaseModel):
    path: str


class ExperienceBuild(BaseModel):
    description: str
    repo: ImageRepo
    version_tag_prefix: str
    system: str
    branch: str
    version: str
    build_command: BuildCommand


class MetricsBuild(BaseModel):
    name: str
    repo: ImageRepo
    version_tag_prefix: str
    systems: list[str]
    version: str
    build_command: BuildCommand


class Builds(BaseModel):
    project: str
    registries: dict[str, ImageRegistry]
    resim_app_config: ResimAppConfig
    experience_build_configs: dict[str, ExperienceBuild]
    metrics_build_configs: dict[str, MetricsBuild]
