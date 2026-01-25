### Data collection
- [ ] Data cleaning and validation
    - [x] Make sure every game in season accounted for
    - [ ] Prune games with overtime
- [x] Get play by play ingestion
- [ ] Make sure PbP scores match team level scores
- [X] Drop teams with < _n_ # obs

### Models
- [x] Table stakes model -- harden this training pipeline
    - Offensive, defensive, and home effect
- [x] Add team-level dispersion
- [ ] Add game-level dispersion
- [x] Clean up logging
- [ ] Model ensemble
- [ ] Model "goodness" metrics

### GUI
- [x] display the parameters for a team
- [x] display the outcomes of the game as simulated in the parquet file
- [x] get rid of ties
- [x] display monte carlo offense/defense ratings

### Infra
- [ ] Switch to AWS S3 for storing simulations

### Tests
- [ ] Test new Modal pipeline
- [ ] Test TeamVolGameRandomness