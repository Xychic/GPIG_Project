CREATE TABLE public.node_to_data_links (
    data_id bigint NOT NULL,
    node_id bigint NOT NULL
);


--
-- TOC entry 217 (class 1259 OID 16422)
-- Name: node_to_node_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.node_to_node_links (
    node_1_id bigint NOT NULL,
    node_2_id bigint NOT NULL,
    weight numeric NOT NULL
);


--
-- TOC entry 216 (class 1259 OID 16415)
-- Name: nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.nodes (
    id bigserial NOT NULL,
    lat numeric NOT NULL,
    lon numeric NOT NULL,
    site_id bigint NOT NULL
);


--
-- TOC entry 218 (class 1259 OID 16437)
-- Name: plant_to_node_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.node_to_plant_links (
    plant_id bigint NOT NULL,
    node_id bigint NOT NULL,
    heading numeric NOT NULL
);


--
-- TOC entry 214 (class 1259 OID 16390)
-- Name: plants; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plants (
    id bigserial NOT NULL,
    species_id bigint NOT NULL,
    diseased boolean,
    date_recorded timestamp
);


--
-- TOC entry 219 (class 1259 OID 16454)
-- Name: sensor_data; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sensor_data (
    id bigserial NOT NULL,
    co2_level numeric,
    ozone_level numeric,
    temperature numeric,
    humidity numeric,
    co_level numeric,
    so2_level numeric,
    no2_level numeric,
    soil_moisture_level numeric,
    soil_temperature numeric,
    soil_humidity numeric,
    soil_ph numeric,
    date_recorded timestamp NOT NULL,
    anomalous boolean
);


--
-- TOC entry 221 (class 1259 OID 16476)
-- Name: sites; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sites (
    id bigserial NOT NULL,
    site_name character varying NOT NULL
);


--
-- TOC entry 215 (class 1259 OID 16394)
-- Name: species; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.species (
    id bigserial NOT NULL,
    species_name character varying NOT NULL
);


--
-- TOC entry 3426 (class 0 OID 16459)
-- Dependencies: 220
-- Data for Name: data_to_node_links; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3423 (class 0 OID 16422)
-- Dependencies: 217
-- Data for Name: node_to_node_links; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3422 (class 0 OID 16415)
-- Dependencies: 216
-- Data for Name: nodes; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3424 (class 0 OID 16437)
-- Dependencies: 218
-- Data for Name: plant_to_node_links; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3420 (class 0 OID 16390)
-- Dependencies: 214
-- Data for Name: plants; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3425 (class 0 OID 16454)
-- Dependencies: 219
-- Data for Name: sensor_data; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3427 (class 0 OID 16476)
-- Dependencies: 221
-- Data for Name: sites; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3421 (class 0 OID 16394)
-- Dependencies: 215
-- Data for Name: species; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- TOC entry 3265 (class 2606 OID 16421)
-- Name: nodes nodes_pk; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nodes
    ADD CONSTRAINT nodes_pk PRIMARY KEY (id);


--
-- TOC entry 3261 (class 2606 OID 16402)
-- Name: plants plants_pk; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plants
    ADD CONSTRAINT plants_pk PRIMARY KEY (id);


--
-- TOC entry 3269 (class 2606 OID 16482)
-- Name: sites sites_pk; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sites
    ADD CONSTRAINT sites_pk PRIMARY KEY (id);


--
-- TOC entry 3267 (class 2606 OID 16465)
-- Name: sensor_data soil_samples_pk; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sensor_data
    ADD CONSTRAINT soil_samples_pk PRIMARY KEY (id);


--
-- TOC entry 3263 (class 2606 OID 16409)
-- Name: species species_pk; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.species
    ADD CONSTRAINT species_pk PRIMARY KEY (id);


ALTER TABLE public.node_to_data_links
    ADD CONSTRAINT node_to_data_links_pk PRIMARY KEY (node_id,data_id);


--
-- TOC entry 3272 (class 2606 OID 16427)
-- Name: node_to_node_links node_links_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.node_to_node_links
    ADD CONSTRAINT node_links_fk FOREIGN KEY (node_1_id) REFERENCES public.nodes(id);


--
-- TOC entry 3273 (class 2606 OID 16432)
-- Name: node_to_node_links node_links_fk_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.node_to_node_links
    ADD CONSTRAINT node_links_fk_1 FOREIGN KEY (node_2_id) REFERENCES public.nodes(id);
   


--
-- TOC entry 3271 (class 2606 OID 16483)
-- Name: nodes nodes_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nodes
    ADD CONSTRAINT nodes_fk FOREIGN KEY (site_id) REFERENCES public.sites(id);


--
-- TOC entry 3274 (class 2606 OID 16442)
-- Name: plant_to_node_links plant_to_node_links_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.node_to_plant_links
    ADD CONSTRAINT plant_to_node_links_fk FOREIGN KEY (node_id) REFERENCES public.nodes(id);


--
-- TOC entry 3275 (class 2606 OID 16447)
-- Name: plant_to_node_links plant_to_node_links_fk_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.node_to_plant_links
    ADD CONSTRAINT plant_to_node_links_fk_1 FOREIGN KEY (plant_id) REFERENCES public.plants(id);


--
-- TOC entry 3270 (class 2606 OID 16410)
-- Name: plants plants_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plants
    ADD CONSTRAINT plants_fk FOREIGN KEY (species_id) REFERENCES public.species(id);


--
-- TOC entry 3276 (class 2606 OID 16466)
-- Name: data_to_node_links soil_to_node_links_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.node_to_data_links
    ADD CONSTRAINT soil_to_node_links_fk FOREIGN KEY (node_id) REFERENCES public.nodes(id);


--
-- TOC entry 3277 (class 2606 OID 16471)
-- Name: data_to_node_links soil_to_node_links_fk_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.node_to_data_links
    ADD CONSTRAINT soil_to_node_links_fk_1 FOREIGN KEY (data_id) REFERENCES public.sensor_data(id);


-- Completed on 2023-11-08 14:19:36 GMT

--
-- PostgreSQL database dump complete
--
