# AYL Front 2 — Codex Operating Rules

## Role

This repo supports Amplify Your Language Front 2: Voice, Character & Lip Sync.

Front 2 is responsible for:
- TTS / voice generation support
- character reference organization
- wardrobe prompt generation and metadata
- video-base test organization
- lip-sync pipeline support
- Front 2 manifests for handoff to Front 3

## Governance

Front 1 is the official documentation authority.

Front 2 must not directly alter official AYL documentation files in:

/Users/fernandoreisdasilva/Library/CloudStorage/GoogleDrive-fernandoreisdasilva@gmail.com/Meu Drive/AYL_Production/00_project_sources/active_documents/

If a documentation change is needed, create a proposal text using:

PROPOSTA DE ALTERAÇÃO

Origem: Frente 2

Problema observado:
...

Impacto na Frente 2:
...

Alteração proposta:
...

Arquivos ou contratos afetados:
...

Exemplo técnico:
...

## Production Root

Default production rooDefault production rooDefault production rooDefragDefault production rooreDefault production rooDefault productiuction

Do not store heavy production media in this Git repo.

Heavy assets belong in GoogleHeavy assets belong in p4
- - - - - - - - - - - - - - - - - w- - - - - - - - - - - - - - - al pr- - - - - - - - - - - - - - - -roduction/0- - - - - - - - - - - - - - -me- - - - - - - - -eference im- - - - - - - - - - - - - - - - -project_sou- - - - - - er_- - - - - - - - - - - - - - - - - wiles for context, but should not modify them.

## Current Test Job

Use this test job for initial wardrobe and Wan 2.7 silent idle validation:

AYL_Production/04_video_jobs/TEST_WARDROBE_WAN_0001/

Purpose:
Validate character wardrobe variation and Wan 2.7 image-to-video silent idle clips before later lip-sync pipeline work.

Wan 2.7 is approved in this repo only for V1 silent visual idle clips without audio. It is not approved here for lip-sync, audio-driven video, spoken character clips, or final scaled production. Replicate remains a test harness; future scaled production is expected to move to RunPod API.

## Visual Rule

One video = one dominant Visual Package.

For TEST_WARDROBE_WAN_0001:
- visual_package: clean-card-grammar
- thumbnail_style: host-right-grammar-contrast
- primary_character: luca

WardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardrobWardroidentity.

Allowed variation:
- outfit
- pose
- expression
- lighting
- background
- crop

Not allowed:
- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- changin- c- notes

Use only officiUse only officiUse only officiUse only officiUse only officiUs_review
approved
blocked
returned_for_revision
final
archived
