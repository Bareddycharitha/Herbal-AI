export type Herb = {
  name: string;
  botanical_name?: string;
  efficacy?: number;
  evidence_level?: string;
  benefits?: string[];
  preparation_method?: string;
  side_effects?: string[];
  contraindications?: string[];
  weight?: number;
};

export type User = {
  id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
};

export type AuthTokens = {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type: string;
};

export type LoginCredentials = {
  email: string;
  password: string;
};

export type RegisterCredentials = {
  email: string;
  password: string;
  full_name?: string;
};

export type HerbDetails = Record<
  string,
  {
    benefits?: string[];
    preparation_method?: string;
    side_effects?: string[];
    contraindications?: string[];
  }
>;

export type DiseaseInformation = {
  description?: string;
  symptoms?: string[];
  causes?: string[];
  prevention?: string[];
  risk_factors?: string[];
  when_to_consult_doctor?: string;
  medical_disclaimer?: string;
};

export type PredictionResponse = {
  success: boolean;
  message: string;
  prediction_id?: string;
  prediction?: {
    disease: string;
    confidence: number;
    confidence_level?: string;
  };
  top_predictions?: { disease: string; confidence: number }[];
  gradcam_image?: string | null;
  disease_information?: DiseaseInformation | null;
  recommended_herbs?: Herb[];
  herb_details?: Record<string, HerbDetails[string]>;
  ai_summary?: string | null;
  skin_ratio?: number;
  edge_ratio?: number;
  recommendation_level?: string;
  warning?: string;
};

export type HerbPredictionResponse = {
  success: boolean;
  message: string;
  prediction?: {
    herb: string;
    confidence: number;
    confidence_level?: string;
  };
  top_predictions?: { herb: string; confidence: number }[];
  herb_information?: {
    common_name?: string;
    scientific_name?: string;
    family?: string;
    description?: string;
    medicinal_properties?: string[];
    traditional_uses?: string[];
    preparation_methods?: string[];
    skin_conditions_supported?: string[];
    active_compounds?: string[];
    precautions?: string[];
    dosage_notes?: string;
    storage_information?: string;
    herb_image?: string | null;
  };
};

export type SummaryResponse = {
  success: boolean;
  summary: string;
  error?: string;
};

export type ChatResponse = {
  success: boolean;
  answer: string;
};
